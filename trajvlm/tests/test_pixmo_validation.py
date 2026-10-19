import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import pyarrow as pa
from datasets import Dataset, DatasetDict, load_from_disk
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_pixmo_validation import prepare


class PixmoValidationTests(unittest.TestCase):
    def test_heldout_images_are_copied_and_train_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'images'
            images.mkdir()
            def image_path(url):
                return images / hashlib.sha256(url.encode()).hexdigest()
            for url in ('train', 'a', 'b'):
                Image.new('RGB', (8, 8)).save(image_path(url), format='PNG')
            image_path('a').chmod(0o600)
            image_path('broken').write_text('invalid image')
            source = Dataset.from_dict(dict(image_url=['train', 'broken', 'a', 'a', 'b'],
                                           caption=['train', 'broken', 'a', 'duplicate', 'b'],
                                           transcripts=[['t']] * 5))
            train = source.select([0]).add_column('image', [str(image_path('train'))])
            DatasetDict(train=train).save_to_disk(str(root / 'train'))
            with pa.OSFile(str(root / 'pixmo-cap-train.arrow'), 'wb') as sink:
                with pa.ipc.new_stream(sink, source.data.schema) as writer:
                    writer.write_table(source.data.table)
            report = prepare(root / 'train', root, images, root / 'output', root / 'eval-images', count=2)
            result = load_from_disk(str(root / 'output'))
            self.assertEqual(result['train'].to_dict(), train.to_dict())
            self.assertEqual(result['validation']['image_url'], ['a', 'b'])
            self.assertEqual(report['overlap_urls'], 0)
            self.assertEqual(report['invalid_candidates_skipped'], 1)
            for url, path in zip(result['validation']['image_url'], result['validation']['image']):
                self.assertEqual(Path(path).read_bytes(), image_path(url).read_bytes())
                self.assertNotEqual(Path(path), image_path(url))
                self.assertEqual(Path(path).stat().st_mode & 0o444, 0o444)
            with self.assertRaises(FileExistsError):
                prepare(root / 'train', root, images, root / 'output', root / 'eval-images', count=2)


if __name__ == '__main__':
    unittest.main()

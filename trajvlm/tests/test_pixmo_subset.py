import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

from datasets import Dataset, load_from_disk
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_pixmo_subset import prepare


class PixmoSubsetTests(unittest.TestCase):
    def test_matching_deduplication_and_safe_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'images'
            images.mkdir()
            for url in ['a', 'b']:
                Image.new('RGB', (8, 8)).save(images / hashlib.sha256(url.encode()).hexdigest(), format='PNG')
            source = Dataset.from_dict(dict(image_url=['a', 'unused', 'a', 'b'],
                                           caption=['first', 'unused', 'duplicate', 'second'],
                                           transcripts=[['t1'], ['t'], ['t2'], ['t3']]))
            with source.formatted_as('arrow'):
                import pyarrow as pa
                with pa.OSFile(str(root / 'pixmo-cap-train.arrow'), 'wb') as sink:
                    with pa.ipc.new_stream(sink, source.data.schema) as writer:
                        writer.write_table(source.data.table)
            output = root / 'prepared' / 'cap'
            report = prepare(root, images, output, expected_count=2)
            result = load_from_disk(str(output))['train']
            self.assertEqual(result['caption'], ['first', 'second'])
            self.assertEqual(report['duplicate_rows_skipped'], 1)
            self.assertEqual(len(result), 2)
            self.assertTrue(all(Path(p).is_file() for p in result['image']))
            with self.assertRaises(FileExistsError):
                prepare(root, images, output, expected_count=2)


if __name__ == '__main__':
    unittest.main()

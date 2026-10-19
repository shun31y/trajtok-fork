"""Save prepared PixMo records with image paths for another disk or host."""
import argparse
import json
from pathlib import Path

from datasets import DatasetDict, load_from_disk


def relocate(source, output, train_images, validation_images=None):
    source, output = Path(source), Path(output)
    staging = output.with_name(output.name + '.incomplete')
    if output.exists() or staging.exists():
        raise FileExistsError(f'Output or staging directory already exists: {output}')
    dataset = load_from_disk(str(source))
    image_dirs = {'train': Path(train_images).resolve()}
    if validation_images is not None:
        image_dirs['validation'] = Path(validation_images).resolve()
    result = {}
    for split, records in dataset.items():
        directory = image_dirs[split]
        paths = [str(directory / Path(p).name) for p in records['image']]
        result[split] = records.remove_columns('image').add_column('image', paths)
    DatasetDict(result).save_to_disk(str(staging))
    report = dict(source=str(source.resolve()), output=str(output.resolve()),
                  examples={k: len(v) for k, v in result.items()},
                  image_directories={k: str(v) for k, v in image_dirs.items()})
    (staging / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    staging.rename(output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'train-images'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--validation-images', type=Path)
    args = parser.parse_args()
    relocate(args.source, args.output, args.train_images, args.validation_images)

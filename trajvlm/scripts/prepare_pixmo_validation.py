"""Copy held-out cached PixMo images and create train + validation records."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from PIL import Image


def prepare(train_dataset, metadata_dir, image_dir, output, validation_images, count=2000):
    train_dataset, metadata_dir, image_dir, output, validation_images = map(
        Path, (train_dataset, metadata_dir, image_dir, output, validation_images))
    staging = output.with_name(output.name + '.incomplete')
    image_staging = validation_images.with_name(validation_images.name + '.incomplete')
    for path in (output, staging, validation_images, image_staging):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')
    if count <= 0:
        raise ValueError('count must be positive')
    train = load_from_disk(str(train_dataset))['train']
    train_urls = set(train['image_url'])
    train_names = {Path(p).name for p in train['image']}
    shards = sorted(metadata_dir.glob('pixmo-cap-train*.arrow'))
    if not shards:
        raise FileNotFoundError(f'No PixMoCap metadata shards in {metadata_dir}')
    source = concatenate_datasets([Dataset.from_file(str(p)) for p in shards])
    seen, indices, sources, names = set(), [], [], []
    invalid = 0
    for i, url in enumerate(source['image_url']):
        key = hashlib.sha256(url.encode('utf-8')).hexdigest()
        if url in train_urls or key in train_names or url in seen:
            continue
        seen.add(url)
        path = image_dir / key
        if not path.is_file():
            continue
        row = source[i]
        if not row['caption'] or not row['transcripts'] or not all(row['transcripts']):
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            invalid += 1
            continue
        indices.append(i)
        sources.append(path)
        names.append(key)
        if len(indices) == count:
            break
    if len(indices) != count:
        raise ValueError(f'Only {len(indices)} / {count} valid held-out images available')
    image_staging.mkdir(parents=True)
    for path, name in zip(sources, names):
        target = image_staging / name
        shutil.copy2(path, target)
        # Downloaded cache files may be owner-only. Published data must also
        # be readable when another server maps the source UID differently.
        target.chmod(stat.S_IMODE(target.stat().st_mode) | 0o444)
    validation = source.select(indices)
    if 'image' in validation.column_names:
        validation = validation.remove_columns('image')
    validation = validation.add_column('image', [str(validation_images.resolve() / n) for n in names])
    assert not train_urls.intersection(validation['image_url'])
    DatasetDict(train=train, validation=validation).save_to_disk(str(staging))
    report = dict(train_examples=len(train), validation_examples=len(validation),
                  overlap_urls=0, verified_copied_images=len(names), invalid_candidates_skipped=invalid,
                  train_source=str(train_dataset.resolve()), metadata_source=str(metadata_dir.resolve()),
                  image_source=str(image_dir.resolve()), validation_images=str(validation_images.resolve()),
                  selection='first valid unique URLs in cached shard order, excluding train URLs and filenames')
    (staging / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    image_staging.rename(validation_images)
    staging.rename(output)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train-dataset', 'metadata-dir', 'image-dir', 'output', 'validation-images'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--count', type=int, default=2000)
    args = parser.parse_args()
    prepare(args.train_dataset, args.metadata_dir, args.image_dir, args.output, args.validation_images, args.count)

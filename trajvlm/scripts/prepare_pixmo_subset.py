"""Prepare PixMoCap records matching an existing URL-hashed image directory.

Reads cached Arrow metadata without network access. Never overwrites output.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets
from PIL import Image


def prepare(metadata_dir, image_dir, output, expected_count=100000, workers=8):
    metadata_dir, image_dir, output = map(Path, (metadata_dir, image_dir, output))
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    shards = sorted(metadata_dir.glob('pixmo-cap-train*.arrow'))
    if not shards:
        raise FileNotFoundError(f'No PixMoCap Arrow shards in {metadata_dir}')
    images = {p.name: p.resolve() for p in image_dir.iterdir() if p.is_file()}
    if len(images) != expected_count:
        raise ValueError(f'Expected {expected_count} image files, found {len(images)}')
    source = concatenate_datasets([Dataset.from_file(str(p)) for p in shards])
    required = {'image_url', 'caption', 'transcripts'}
    if not required.issubset(source.column_names):
        raise ValueError(f'Missing metadata columns: {required - set(source.column_names)}')
    indices, paths, matched = [], [], set()
    duplicate_rows = 0
    for i, url in enumerate(source['image_url']):
        key = hashlib.sha256(url.encode('utf-8')).hexdigest()
        if key in images:
            if key in matched:
                # One record per image, deterministic in cached shard order.
                duplicate_rows += 1
                continue
            matched.add(key)
            indices.append(i)
            paths.append(str(images[key]))
    if len(matched) != expected_count:
        raise ValueError(f'Only {len(matched)} / {expected_count} images match metadata')
    selected = source.select(indices)
    for row in selected.select_columns(['caption', 'transcripts']):
        if not row['caption'] or not row['transcripts'] or not all(row['transcripts']):
            raise ValueError('Missing caption/transcript in selected records')

    def verify(path):
        with Image.open(path) as image:
            image.verify()

    print(f'Verifying {len(paths)} images...', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(verify, paths):
            pass
    if 'image' in selected.column_names:
        selected = selected.remove_columns('image')
    selected = selected.add_column('image', paths)
    temporary = output.with_name(output.name + '.incomplete')
    if temporary.exists():
        raise FileExistsError(f'Refusing to overwrite staging directory {temporary}')
    DatasetDict(train=selected).save_to_disk(str(temporary))
    report = dict(train_examples=len(selected), image_dir=str(image_dir.resolve()),
                  metadata_dir=str(metadata_dir.resolve()),
                  metadata_shards=[p.name for p in shards], verified_images=len(paths),
                  duplicate_rows_skipped=duplicate_rows,
                  duplicate_policy='first record per image URL in shard order',
                  columns=selected.column_names)
    (temporary / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    temporary.rename(output)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-dir', type=Path, required=True)
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-count', type=int, default=100000)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    prepare(args.metadata_dir, args.image_dir, args.output, args.expected_count, args.workers)

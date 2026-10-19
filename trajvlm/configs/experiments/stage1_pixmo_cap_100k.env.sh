# Source from the repository root before using the 100k recipe.
source trajvlm/scripts/data_env.sh
export PIXMO_IMAGE_DIR=/mnt/ssd2/shun/VideoUnderstanding/trajtok/torch_datasets/pixmo_images_100k
# A separate prepared dataset prevents accidental training on the full PixMo set.
# Train: 100000 records; validation: 2000 additional held-out records.
export PIXMO_CAP_DATASET=/mnt/ssd2/shun/VideoUnderstanding/trajtok/torch_datasets/pixmo_datasets_100k/cap_with_validation

# Source from the repository root before using the full PixMoCap recipe on /data-premium.
export MOLMO_DATA_DIR=/data/trajtok
source trajvlm/scripts/data_env.sh
export PIXMO_DATA_DIR=/data-premium/PixMoCap
export PIXMO_IMAGE_DIR=/data-premium/PixMoCap/torch_datasets/pixmo_images
# Records relocated from pixmo_datasets/cap (whose image paths point at /mnt/ssd2)
# with scripts/relocate_pixmo_dataset.py. Train: 706022 records; validation: 2048.
export PIXMO_CAP_DATASET=/data-premium/PixMoCap/torch_datasets/pixmo_datasets/cap_data_premium
export WANDB_PROJECT=trajvlm_stage1
export WANDB_ENTITY=odango_team
set -eux

## Settings
EXPERIMENT_NAME="SciGA_interGA_RandomSampling"
SEED=42
DEVICE=0
DATASET_JSON_DIR="./SciGA_for_experiments/json/"
DATASET_FIGURE_DIR="./SciGA_for_experiments/figures/"
SAVE_CACHE_DIR="./benchmark/output/caches/"
BATCH_SIZE=8
IS_WANDB=false

## Create command
CMD="poetry run python ./benchmark/experiments/train_inter_with_random.py \
  --experiment_name $EXPERIMENT_NAME \
  --seed $SEED \
  --device $DEVICE \
  --dataset_json_dir $DATASET_JSON_DIR \
  --dataset_figure_dir $DATASET_FIGURE_DIR \
  --save_cache_dir $SAVE_CACHE_DIR \
  --batch_size $BATCH_SIZE"

if [ "$IS_WANDB" = true ]; then
  CMD="$CMD --is_wandb"
fi

## Run the command
$CMD

set -eux

## Settings
EXPERIMENT_NAME="SciGA_interGA_Abs2Fig"
SEED=42
DEVICE=0
DATASET_JSON_DIR="./SciGA_for_experiments/json/"
DATASET_FIGURE_DIR="./SciGA_for_experiments/figures/"
SAVE_CHECKPOINT_DIR="./benchmark/output/checkpoints/"
SAVE_CACHE_DIR="./benchmark/output/caches/"
MODEL_TYPE="CLIP"
MODEL_NAME=""
MODEL_CONFIG_PATH=""
WEIGHT_DECAY=1e-3
EPOCHS=15
NUM_WORKERS=2
BATCH_SIZE=8
ACCUM_ITER=128
LEARNING_RATE=1e-6
IS_MERGE_CAPTION=false
IS_WANDB=false

## Create command
CMD="poetry run python ./benchmark/experiments/train_inter_with_abs2fig.py \
  --experiment_name $EXPERIMENT_NAME \
  --seed $SEED \
  --device $DEVICE \
  --dataset_json_dir $DATASET_JSON_DIR \
  --dataset_figure_dir $DATASET_FIGURE_DIR \
  --save_checkpoint_dir $SAVE_CHECKPOINT_DIR \
  --save_cache_dir $SAVE_CACHE_DIR \
  --model_type $MODEL_TYPE \
  --weight_decay $WEIGHT_DECAY \
  --epochs $EPOCHS \
  --num_workers $NUM_WORKERS \
  --batch_size $BATCH_SIZE \
  --accum_iter $ACCUM_ITER \
  --learning_rate $LEARNING_RATE"

if [ "$MODEL_NAME" != "" ]; then
  CMD="$CMD --model_name $MODEL_NAME"
fi

if [ "$MODEL_CONFIG_PATH" != "" ]; then
  CMD="$CMD --model_config_path $MODEL_CONFIG_PATH"
fi

if [ "$IS_MERGE_CAPTION" = true ]; then
  CMD="$CMD --is_merge_caption"
fi

if [ "$IS_WANDB" = true ]; then
  CMD="$CMD --is_wandb"
fi

## Run the command
$CMD

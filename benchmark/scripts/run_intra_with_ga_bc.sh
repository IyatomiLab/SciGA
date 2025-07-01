set -eux

## Settings
EXPERIMENT_NAME="SciGA_intraGA_GA-BC"
SEED=42
DEVICE=0
DATASET_JSON_DIR="./SciGA_for_experiments/json/"
DATASET_FIGURE_DIR="./SciGA_for_experiments/figures/"
SAVE_CHECKPOINT_DIR="./benchmark/output/checkpoints/"
MODEL_TYPE="CLIP"
MODEL_NAME=""
WEIGHT_DECAY=1e-3
EPOCHS=15
NUM_WORKERS=2
BATCH_SIZE=8
ACCUM_ITER=128
LEARNING_RATE=1e-6
IS_WANDB=false

## Create command
CMD="poetry run python ./benchmark/experiments/train_intra_with_ga_bc.py \
  --experiment_name $EXPERIMENT_NAME \
  --seed $SEED \
  --device $DEVICE \
  --dataset_json_dir $DATASET_JSON_DIR \
  --dataset_figure_dir $DATASET_FIGURE_DIR \
  --save_checkpoint_dir $SAVE_CHECKPOINT_DIR \
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

if [ "$IS_WANDB" = true ]; then
  CMD="$CMD --is_wandb"
fi

## Run the command
$CMD

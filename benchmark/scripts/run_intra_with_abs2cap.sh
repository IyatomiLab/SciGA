set -eux

## Settings
EXPERIMENT_NAME="SciGA_intraGA_Abs2Cap"
SEED=42
DATASET_JSON_DIR="./SciGA_for_experiments/json/"
MODEL_TYPE="ROUGE"
IS_WANDB=false

## Create command
CMD="poetry run python ./benchmark/experiments/train_intra_with_abs2cap.py \
  --experiment_name $EXPERIMENT_NAME \
  --seed $SEED \
  --dataset_json_dir $DATASET_JSON_DIR \
  --model_type $MODEL_TYPE"

if [ "$IS_WANDB" = true ]; then
  CMD="$CMD --is_wandb"
fi

## Run the command
$CMD

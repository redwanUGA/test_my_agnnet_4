#!/bin/bash
# Install dependencies, download datasets and run all model-dataset experiments.
set -e

# 1. Install Python requirements
pip install -r requirements.txt

# 2. Download datasets if missing
if [ ! -d "simple_data" ]; then
  echo "Downloading datasets..."
  gdown 'https://drive.google.com/drive/folders/1iZE_Cg5wAk_94Uk1DgNrOLiqp4F6cbfZ?usp=sharing' --folder
fi

SEARCH_EPOCHS=1
EPOCHS=2
SAVE_DIR=saved_models
mkdir -p "$SAVE_DIR"

models=(BaselineGCN GraphSAGE GAT TGAT TGN AGNNet)
datasets=("OGB-Arxiv" "Reddit" "TGB-Wiki" "MOOC")

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    model_file="$SAVE_DIR/${model}_${dataset}.pt"
    config_file="$SAVE_DIR/${model}_${dataset}_params.json"
    if [ ! -f "$model_file" ] || [ ! -f "$config_file" ]; then
      python hyperparameter_search.py --model "$model" --dataset "$dataset" --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
    fi
    python main.py --model "$model" --dataset "$dataset" --epochs $EPOCHS --load-model "$model_file" --config "$config_file"
  done
done

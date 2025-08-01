#!/bin/bash
# Run all model and dataset combinations sequentially without loops.
# Assumes datasets are located in the 'simple_data/' folder.

SEARCH_EPOCHS=1
EPOCHS=2
SAVE_DIR=saved_models

# BaselineGCN
if [ ! -f "$SAVE_DIR/BaselineGCN_OGB-Arxiv.pt" ] || [ ! -f "$SAVE_DIR/BaselineGCN_OGB-Arxiv_params.json" ]; then
    python hyperparameter_search.py --model BaselineGCN --dataset OGB-Arxiv --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model BaselineGCN --dataset OGB-Arxiv --epochs $EPOCHS --load-model "$SAVE_DIR/BaselineGCN_OGB-Arxiv.pt" --config "$SAVE_DIR/BaselineGCN_OGB-Arxiv_params.json"

if [ ! -f "$SAVE_DIR/BaselineGCN_Reddit.pt" ] || [ ! -f "$SAVE_DIR/BaselineGCN_Reddit_params.json" ]; then
    python hyperparameter_search.py --model BaselineGCN --dataset Reddit --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model BaselineGCN --dataset Reddit --epochs $EPOCHS --load-model "$SAVE_DIR/BaselineGCN_Reddit.pt" --config "$SAVE_DIR/BaselineGCN_Reddit_params.json"

if [ ! -f "$SAVE_DIR/BaselineGCN_TGB-Wiki.pt" ] || [ ! -f "$SAVE_DIR/BaselineGCN_TGB-Wiki_params.json" ]; then
    python hyperparameter_search.py --model BaselineGCN --dataset TGB-Wiki --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model BaselineGCN --dataset TGB-Wiki --epochs $EPOCHS --load-model "$SAVE_DIR/BaselineGCN_TGB-Wiki.pt" --config "$SAVE_DIR/BaselineGCN_TGB-Wiki_params.json"

if [ ! -f "$SAVE_DIR/BaselineGCN_MOOC.pt" ] || [ ! -f "$SAVE_DIR/BaselineGCN_MOOC_params.json" ]; then
    python hyperparameter_search.py --model BaselineGCN --dataset MOOC --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model BaselineGCN --dataset MOOC --epochs $EPOCHS --load-model "$SAVE_DIR/BaselineGCN_MOOC.pt" --config "$SAVE_DIR/BaselineGCN_MOOC_params.json"

# GraphSAGE
if [ ! -f "$SAVE_DIR/GraphSAGE_OGB-Arxiv.pt" ] || [ ! -f "$SAVE_DIR/GraphSAGE_OGB-Arxiv_params.json" ]; then
    python hyperparameter_search.py --model GraphSAGE --dataset OGB-Arxiv --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model GraphSAGE --dataset OGB-Arxiv --epochs $EPOCHS --load-model "$SAVE_DIR/GraphSAGE_OGB-Arxiv.pt" --config "$SAVE_DIR/GraphSAGE_OGB-Arxiv_params.json"

if [ ! -f "$SAVE_DIR/GraphSAGE_Reddit.pt" ] || [ ! -f "$SAVE_DIR/GraphSAGE_Reddit_params.json" ]; then
    python hyperparameter_search.py --model GraphSAGE --dataset Reddit --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model GraphSAGE --dataset Reddit --epochs $EPOCHS --load-model "$SAVE_DIR/GraphSAGE_Reddit.pt" --config "$SAVE_DIR/GraphSAGE_Reddit_params.json"

if [ ! -f "$SAVE_DIR/GraphSAGE_TGB-Wiki.pt" ] || [ ! -f "$SAVE_DIR/GraphSAGE_TGB-Wiki_params.json" ]; then
    python hyperparameter_search.py --model GraphSAGE --dataset TGB-Wiki --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model GraphSAGE --dataset TGB-Wiki --epochs $EPOCHS --load-model "$SAVE_DIR/GraphSAGE_TGB-Wiki.pt" --config "$SAVE_DIR/GraphSAGE_TGB-Wiki_params.json"

if [ ! -f "$SAVE_DIR/GraphSAGE_MOOC.pt" ] || [ ! -f "$SAVE_DIR/GraphSAGE_MOOC_params.json" ]; then
    python hyperparameter_search.py --model GraphSAGE --dataset MOOC --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model GraphSAGE --dataset MOOC --epochs $EPOCHS --load-model "$SAVE_DIR/GraphSAGE_MOOC.pt" --config "$SAVE_DIR/GraphSAGE_MOOC_params.json"

# GAT
if [ ! -f "$SAVE_DIR/GAT_OGB-Arxiv.pt" ] || [ ! -f "$SAVE_DIR/GAT_OGB-Arxiv_params.json" ]; then
    python hyperparameter_search.py --model GAT --dataset OGB-Arxiv --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model GAT --dataset OGB-Arxiv --epochs $EPOCHS --load-model "$SAVE_DIR/GAT_OGB-Arxiv.pt" --config "$SAVE_DIR/GAT_OGB-Arxiv_params.json"

if [ ! -f "$SAVE_DIR/GAT_Reddit.pt" ] || [ ! -f "$SAVE_DIR/GAT_Reddit_params.json" ]; then
    python hyperparameter_search.py --model GAT --dataset Reddit --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model GAT --dataset Reddit --epochs $EPOCHS --load-model "$SAVE_DIR/GAT_Reddit.pt" --config "$SAVE_DIR/GAT_Reddit_params.json"

if [ ! -f "$SAVE_DIR/GAT_TGB-Wiki.pt" ] || [ ! -f "$SAVE_DIR/GAT_TGB-Wiki_params.json" ]; then
    python hyperparameter_search.py --model GAT --dataset TGB-Wiki --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model GAT --dataset TGB-Wiki --epochs $EPOCHS --load-model "$SAVE_DIR/GAT_TGB-Wiki.pt" --config "$SAVE_DIR/GAT_TGB-Wiki_params.json"

if [ ! -f "$SAVE_DIR/GAT_MOOC.pt" ] || [ ! -f "$SAVE_DIR/GAT_MOOC_params.json" ]; then
    python hyperparameter_search.py --model GAT --dataset MOOC --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model GAT --dataset MOOC --epochs $EPOCHS --load-model "$SAVE_DIR/GAT_MOOC.pt" --config "$SAVE_DIR/GAT_MOOC_params.json"

# TGAT
if [ ! -f "$SAVE_DIR/TGAT_OGB-Arxiv.pt" ] || [ ! -f "$SAVE_DIR/TGAT_OGB-Arxiv_params.json" ]; then
    python hyperparameter_search.py --model TGAT --dataset OGB-Arxiv --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model TGAT --dataset OGB-Arxiv --epochs $EPOCHS --load-model "$SAVE_DIR/TGAT_OGB-Arxiv.pt" --config "$SAVE_DIR/TGAT_OGB-Arxiv_params.json"

if [ ! -f "$SAVE_DIR/TGAT_Reddit.pt" ] || [ ! -f "$SAVE_DIR/TGAT_Reddit_params.json" ]; then
    python hyperparameter_search.py --model TGAT --dataset Reddit --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model TGAT --dataset Reddit --epochs $EPOCHS --load-model "$SAVE_DIR/TGAT_Reddit.pt" --config "$SAVE_DIR/TGAT_Reddit_params.json"

if [ ! -f "$SAVE_DIR/TGAT_TGB-Wiki.pt" ] || [ ! -f "$SAVE_DIR/TGAT_TGB-Wiki_params.json" ]; then
    python hyperparameter_search.py --model TGAT --dataset TGB-Wiki --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model TGAT --dataset TGB-Wiki --epochs $EPOCHS --load-model "$SAVE_DIR/TGAT_TGB-Wiki.pt" --config "$SAVE_DIR/TGAT_TGB-Wiki_params.json"

if [ ! -f "$SAVE_DIR/TGAT_MOOC.pt" ] || [ ! -f "$SAVE_DIR/TGAT_MOOC_params.json" ]; then
    python hyperparameter_search.py --model TGAT --dataset MOOC --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model TGAT --dataset MOOC --epochs $EPOCHS --load-model "$SAVE_DIR/TGAT_MOOC.pt" --config "$SAVE_DIR/TGAT_MOOC_params.json"

# TGN
if [ ! -f "$SAVE_DIR/TGN_OGB-Arxiv.pt" ] || [ ! -f "$SAVE_DIR/TGN_OGB-Arxiv_params.json" ]; then
    python hyperparameter_search.py --model TGN --dataset OGB-Arxiv --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model TGN --dataset OGB-Arxiv --epochs $EPOCHS --load-model "$SAVE_DIR/TGN_OGB-Arxiv.pt" --config "$SAVE_DIR/TGN_OGB-Arxiv_params.json"

if [ ! -f "$SAVE_DIR/TGN_Reddit.pt" ] || [ ! -f "$SAVE_DIR/TGN_Reddit_params.json" ]; then
    python hyperparameter_search.py --model TGN --dataset Reddit --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model TGN --dataset Reddit --epochs $EPOCHS --load-model "$SAVE_DIR/TGN_Reddit.pt" --config "$SAVE_DIR/TGN_Reddit_params.json"

if [ ! -f "$SAVE_DIR/TGN_TGB-Wiki.pt" ] || [ ! -f "$SAVE_DIR/TGN_TGB-Wiki_params.json" ]; then
    python hyperparameter_search.py --model TGN --dataset TGB-Wiki --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model TGN --dataset TGB-Wiki --epochs $EPOCHS --load-model "$SAVE_DIR/TGN_TGB-Wiki.pt" --config "$SAVE_DIR/TGN_TGB-Wiki_params.json"

if [ ! -f "$SAVE_DIR/TGN_MOOC.pt" ] || [ ! -f "$SAVE_DIR/TGN_MOOC_params.json" ]; then
    python hyperparameter_search.py --model TGN --dataset MOOC --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model TGN --dataset MOOC --epochs $EPOCHS --load-model "$SAVE_DIR/TGN_MOOC.pt" --config "$SAVE_DIR/TGN_MOOC_params.json"

# AGNNet
if [ ! -f "$SAVE_DIR/AGNNet_OGB-Arxiv.pt" ] || [ ! -f "$SAVE_DIR/AGNNet_OGB-Arxiv_params.json" ]; then
    python hyperparameter_search.py --model AGNNet --dataset OGB-Arxiv --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model AGNNet --dataset OGB-Arxiv --epochs $EPOCHS --load-model "$SAVE_DIR/AGNNet_OGB-Arxiv.pt" --config "$SAVE_DIR/AGNNet_OGB-Arxiv_params.json"

if [ ! -f "$SAVE_DIR/AGNNet_Reddit.pt" ] || [ ! -f "$SAVE_DIR/AGNNet_Reddit_params.json" ]; then
    python hyperparameter_search.py --model AGNNet --dataset Reddit --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model AGNNet --dataset Reddit --epochs $EPOCHS --load-model "$SAVE_DIR/AGNNet_Reddit.pt" --config "$SAVE_DIR/AGNNet_Reddit_params.json"

if [ ! -f "$SAVE_DIR/AGNNet_TGB-Wiki.pt" ] || [ ! -f "$SAVE_DIR/AGNNet_TGB-Wiki_params.json" ]; then
    python hyperparameter_search.py --model AGNNet --dataset TGB-Wiki --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model AGNNet --dataset TGB-Wiki --epochs $EPOCHS --load-model "$SAVE_DIR/AGNNet_TGB-Wiki.pt" --config "$SAVE_DIR/AGNNet_TGB-Wiki_params.json"

if [ ! -f "$SAVE_DIR/AGNNet_MOOC.pt" ] || [ ! -f "$SAVE_DIR/AGNNet_MOOC_params.json" ]; then
    python hyperparameter_search.py --model AGNNet --dataset MOOC --epochs $SEARCH_EPOCHS --save-dir "$SAVE_DIR"
fi
python main.py --model AGNNet --dataset MOOC --epochs $EPOCHS --load-model "$SAVE_DIR/AGNNet_MOOC.pt" --config "$SAVE_DIR/AGNNet_MOOC_params.json"


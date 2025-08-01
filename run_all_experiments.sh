#!/bin/bash
# Run all model and dataset combinations sequentially.
# Assumes datasets are located in the 'simple_data/' folder.

EPOCHS=2

python main.py --model BaselineGCN --dataset OGB-Arxiv --epochs $EPOCHS
python main.py --model BaselineGCN --dataset Reddit --epochs $EPOCHS
python main.py --model BaselineGCN --dataset TGB-Wiki --epochs $EPOCHS
python main.py --model BaselineGCN --dataset MOOC --epochs $EPOCHS

python main.py --model GraphSAGE --dataset OGB-Arxiv --epochs $EPOCHS
python main.py --model GraphSAGE --dataset Reddit --epochs $EPOCHS
python main.py --model GraphSAGE --dataset TGB-Wiki --epochs $EPOCHS
python main.py --model GraphSAGE --dataset MOOC --epochs $EPOCHS

python main.py --model TGAT --dataset OGB-Arxiv --epochs $EPOCHS
python main.py --model TGAT --dataset Reddit --epochs $EPOCHS
python main.py --model TGAT --dataset TGB-Wiki --epochs $EPOCHS
python main.py --model TGAT --dataset MOOC --epochs $EPOCHS

python main.py --model TGN --dataset OGB-Arxiv --epochs $EPOCHS
python main.py --model TGN --dataset Reddit --epochs $EPOCHS
python main.py --model TGN --dataset TGB-Wiki --epochs $EPOCHS
python main.py --model TGN --dataset MOOC --epochs $EPOCHS

python main.py --model AGNNet --dataset OGB-Arxiv --epochs $EPOCHS
python main.py --model AGNNet --dataset Reddit --epochs $EPOCHS
python main.py --model AGNNet --dataset TGB-Wiki --epochs $EPOCHS
python main.py --model AGNNet --dataset MOOC --epochs $EPOCHS


#!/bin/bash
export DATA_DIR='/home/dlf/pyprojects/hw_transformer_re/dataset'
export PretreainModel_DIR='/home/dlf/pyprojects/pretrain_models/bert-base-uncased'
python ../main.py \
       --dataset='tacred' \
       --data_dir=$DATA_DIR/tacred \
       --vocab_dir=$DATA_DIR/vocab \
       --memo='_bert_xavier' \
       --save_dir='../output' \
       --use_bert_embedding \
       --pretrained_mode=$PretreainModel_DIR \
       --max_length=200






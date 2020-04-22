#!/bin/bash
export DATA_DIR='/home/dlf/pyprojects/hw_transformer_re/dataset'
export PretreainModel_DIR='/home/dlf/pyprojects/pretrain_models/bert-base-uncased'
export PROJECT_DIR='/home/dlf/pyprojects/FAT-RE'

python $PROJECT_DIR/utils/prepare_vocab.py \
    --vocab=$DATA_DIR/w2v_vocab \
    --data_dir=$DATA_DIR/tacred \
    --lower \
    --wv_dim=300 \
    --wv_file='/home/dlf/pyprojects/GoogleNews-vectors-negative300.txt'
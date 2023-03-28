#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 -u bert_eval.py \
	--pretrained_model_path saved_models/Laptops/train/best_model.pt_new4.21 \
  --data_dir ../dataset/Biaffine/glove/Laptops \
	--vocab_dir ../dataset/Biaffine/glove/Laptops \

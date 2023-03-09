# -*- coding: utf-8 -*-
# @Time    : 2023/2/16 17:10
# @Author  : huangkai
# @File    : config.py
maxlen = 256
epochs = 10
vocab_dim=256
batch_size = 32
bert_layers = 12
learning_rate = 2e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率
categories = set()
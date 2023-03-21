# -*- coding: utf-8 -*-
# @Time    : 2023/3/12 20:08
# @Author  : huangkai
# @File    : config.py


maxlen = 50
epochs = 20
batch_size = 8
bert_layers = 12
embed_size=128
learning_rate = 2e-3  # bert_layers越小，学习率应该要越大
crf_lr_multiplier =1000
dict_path = '../../uncased_L-12_H-768_A-12/vocab.txt'
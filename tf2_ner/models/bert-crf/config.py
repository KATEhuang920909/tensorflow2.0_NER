# -*- coding: utf-8 -*-
# @Time    : 2023/3/12 20:08
# @Author  : huangkai
# @File    : config.py


maxlen = 256
epochs = 10
batch_size = 32
bert_layers = 12
learning_rate = 2e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率
# bert配置
config_path = '../../uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../../uncased_L-12_H-768_A-12/vocab.txt'

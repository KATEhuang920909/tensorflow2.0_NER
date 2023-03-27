#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/26 上午12:35
@Auth ： huangkai
@File ：config.py
@IDE ：PyCharm
"""
maxlen = 50
epochs = 10
units=128
batch_size = 4
bert_layers = 12
learning_rate = 2e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率
# bert配置
config_path = '../../chinese_bert_wwm_L-12_H-768_A-12/publish/bert_config.json'
checkpoint_path = '../../chinese_bert_wwm_L-12_H-768_A-12/publish/bert_model.ckpt'
dict_path = '../../chinese_bert_wwm_L-12_H-768_A-12/publish/vocab.txt'

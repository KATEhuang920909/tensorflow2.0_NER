#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/23 上午12:38
@Auth ： huangkai
@File ：config.py
@model_name ：biaffine
"""
maxlen = 50
epochs = 10
units=128
batch_size = 4
bert_layers = 12
learning_rate = 2e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率
# bert配置
# config_path = '../../uncased_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '../../uncased_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = '../../uncased_L-12_H-768_A-12/vocab.txt'

config_path = r"D:\代码\pretrained_model\chinese_bert_wwm/bert_config.json"
checkpoint_path = r"D:\代码\pretrained_model\chinese_bert_wwm/bert_model.ckpt"
dict_path = r"D:\代码\pretrained_model\chinese_bert_wwm/vocab.txt"

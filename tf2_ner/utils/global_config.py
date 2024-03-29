# -*- coding: utf-8 -*-
# @Time    : 2023/3/12 22:12
# @Author  : huangkai
# @File    : global_config.py

import os

current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前地址

# 词表目录(本词表采用的是 bert预训练模型的词表)
path_vocab = os.path.join(current_dir, '../data/vocab.txt')

# 实体命名识别文件目录
path_data_dir = os.path.join(current_dir, '../data/data/')
path_data2_dir = os.path.join(current_dir, '../data/data2/')
path_msra_dir = os.path.join(current_dir, '../data/MSRA/')
path_renmin_dir = os.path.join(current_dir, '../data/')

# bert 预训练文件地址
path_bert_dir = os.path.join(r"D:\代码\pretrained_model\uncased_L-12_H-768_A-12")

# 日志、记录类文件目录地址
path_log_dir = os.path.join(current_dir, "../log")

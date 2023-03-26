#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/26 上午12:35
@Auth ： huangkai
@File ：train.py
@IDE ：PyCharm
"""
import pandas as pd
import tensorflow as tf
import sys

sys.path.append("../../")
from model import CascadeNER, forward_step
from config import *
from keras.utils.vis_utils import plot_model
from bert4keras.snippets import sequence_padding
from bert4keras.tokenizers import Tokenizer
import numpy as np
from utils.data_helper import load_dataset

categories = set()
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def load_data(filename, is_test=False):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    if is_test:
        with open(filename, encoding='utf-8') as f:
            data = f.readlines()
            data = [k.strip().split("\x01")[1] for k in data]
        return data if len(data) > 0 else None
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d = ['']
            for i, c in enumerate(l.split('\n')):
                if len(c.strip()) != 0:
                    char, flag = c.split(' ')
                    d[0] += char
                    if flag[0] == 'B':
                        d.append([i, i, flag[2:]])
                        categories.add(flag[2:])
                    elif flag[0] == 'E':
                        d[-1][1] = i
                    elif flag[0] == "O":
                        categories.add("others")

            D.append(d)
    return D


def ner_tokenizers(data):
    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    categories_idx = pd.factorize((categories))[0]

    # one_hot_categories = tf.keras.utils.to_categorical(categories_idx, num_classes=len(categories))
    # print(one_hot_categories)
    for d in data:
        tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        labels = np.zeros(shape=(maxlen, maxlen, len(categories)))
        # print(labels)
        for start, end, label in d[1:]:
            # print(end, start, categories.index(label))
            # print(categories_idx[categories.index(label)])
            # print(labels.shape)
            labels[end][start][categories_idx[categories.index(label)]] = categories_idx[categories.index(label)]  # 下三角
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_labels.append(labels)

    batch_token_ids = sequence_padding(batch_token_ids, length=maxlen)
    batch_segment_ids = sequence_padding(batch_segment_ids, length=maxlen)
    batch_labels = sequence_padding(batch_labels, length=maxlen)
    return {"token_id": batch_token_ids,
            "segment_id": batch_segment_ids,
            "label": batch_labels}  # batch_size,seq_len,seq_len,num_classes


if __name__ == '__main__':

    train_data = load_data('../../data/address/train.conll')
    valid_data = load_data('../../data/address/dev.conll')
    print(len(train_data), len(valid_data))
    # test_data = load_data('../../data/address/final_test.txt', is_test=True)
    categories = list(sorted(categories))
    train_data_token = ner_tokenizers(train_data[:200])
    valid_data_token = ner_tokenizers(valid_data)
    # print(train_data["token_id"].shape)
    train_data_gen, valid_data_gen = load_dataset(train_data_token, valid_data_token, batch_size=batch_size)

    # train_data = data_generator(train_data, batch_size=batch_size)
    BertCrfmodel = CascadeNER(num_classes=len(categories), mask=True)
    BertCrfmodel.build(input_shape={"token_id": [None, maxlen],
                                    "segment_id": [None, maxlen],
                                    "label": [None, maxlen, maxlen, len(categories)]})
    # print(BertCrfmodel.summary())
    # exit()
    # plot_model(BertCrfmodel, to_file='BERT_BILSTM_CRF.png', show_shapes=True)
    optimizer = tf.optimizers.Adam(learning_rate=1e-3)
    train_loss_metric = tf.keras.metrics.Mean()
    train_f1_metric = tf.keras.metrics.Mean()
    valid_loss_metric = tf.keras.metrics.Mean()
    valid_f1_metric = tf.keras.metrics.Mean()
    # print(len(train_data))
    # exit()
    with tf.device("CPU: 0"):
        best_f1_score = 0.0
        for epoch in range(10):
            for i, batch_inputs in enumerate(train_data_gen):
                loss, f1_score = forward_step(batch_inputs, BertCrfmodel, optimizer=optimizer)
                train_loss_metric(loss)
                train_f1_metric(f1_score)
                progress = {
                    "epoch": epoch + 1,
                    "progress": (i + 1) * batch_size if (i + 1) * batch_size < len(train_data) else len(train_data),
                    "BCE_loss": train_loss_metric.result().numpy(),
                    "f1_score": train_f1_metric.result().numpy()}
                print("epoch", progress["epoch"],
                      f"{progress['progress']}/{len(train_data)}",
                      "BCE_loss", progress["BCE_loss"],
                      "f1_score", progress["f1_score"])
            print("epoch:", epoch + 1, "train:", train_loss_metric.result().numpy(), train_f1_metric.result().numpy())
            for batch_inputs in valid_data_gen:
                val_loss, val_f1 = BertCrfmodel(batch_inputs)
                valid_loss_metric(val_loss)
                valid_f1_metric(val_f1)
            valid_f1 = valid_f1_metric.result().numpy()
            if valid_f1 >= best_f1_score:
                best_f1_score = valid_f1
                BertCrfmodel.save_weights("/best_model/best_model.weights")
            print("val:", valid_loss_metric.result().numpy(), valid_f1_metric.result().numpy(),
                  "best_f1_score:", best_f1_score)
            train_loss_metric.reset_states()
            train_f1_metric.reset_states()
            valid_loss_metric.reset_states()
            valid_f1_metric.reset_states()
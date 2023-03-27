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
from cascade_span import CascadeSpan, forward_step
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
    cate_id = dict(zip(categories, np.arange(len(categories))))
    batch_token_ids, batch_segment_ids = [], []
    batch_head_bio_labels, batch_tail_bio_labels, batch_head_cls_labels, batch_tail_cls_labels = [], [], [], []

    for d in data:
        tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        labels = [np.zeros(shape=(maxlen))] * 4
        # print(labels)
        for start, end, label in d[1:]:
            labels[0][start] = 1
            labels[1][end] = 1
            labels[2][start] = cate_id[label]
            labels[3][end] = cate_id[label]
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_head_bio_labels.append(labels[0])
        batch_tail_bio_labels.append(labels[1])
        batch_head_cls_labels.append(labels[2])
        batch_tail_cls_labels.append(labels[3])

    batch_token_ids = sequence_padding(batch_token_ids, length=maxlen)
    batch_segment_ids = sequence_padding(batch_segment_ids, length=maxlen)
    batch_head_bio_labels = sequence_padding(batch_head_bio_labels, length=maxlen)
    batch_tail_bio_labels = sequence_padding(batch_tail_bio_labels, length=maxlen)
    batch_head_cls_labels = sequence_padding(batch_head_cls_labels, length=maxlen)
    batch_tail_cls_labels = sequence_padding(batch_tail_cls_labels, length=maxlen)
    # print(batch_head_bio_labels.shape,batch_tail_bio_labels.shape,batch_head_cls_labels.shape,batch_tail_cls_labels.shape)
    return {"token_id": batch_token_ids,
            "segment_id": batch_segment_ids,
            "head_bio_labels": batch_head_bio_labels,
            "tail_bio_labels": batch_tail_bio_labels,
            "head_cls_labels": batch_head_cls_labels,
            "tail_cls_labels": batch_tail_cls_labels, }  # batch_size,seq_len,seq_len,num_classes


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
    BertCrfmodel = CascadeSpan(num_classes=len(categories), mask=False)
    BertCrfmodel.build(input_shape={"token_id": [batch_size, maxlen],
                                    "segment_id": [batch_size, maxlen],
                                    "head_bio_labels": [batch_size, maxlen],
                                    "tail_bio_labels": [batch_size, maxlen],
                                    "head_cls_labels": [batch_size, maxlen],
                                    "tail_cls_labels": [batch_size, maxlen], })
    print(BertCrfmodel.summary())
    # exit()
    # plot_model(BertCrfmodel, to_file='BERT_BILSTM_CRF.png', show_shapes=True)
    optimizer = tf.optimizers.Adam(learning_rate=1e-3)
    train_loss_metric = tf.keras.metrics.Mean()
    train_f1_metric = [tf.keras.metrics.Mean()] * 4
    valid_loss_metric = tf.keras.metrics.Mean()
    valid_f1_metric = [tf.keras.metrics.Mean()] * 4
    # print(len(train_data))
    # exit()
    with tf.device("CPU: 0"):
        best_f1_score = 0.0
        for epoch in range(10):
            for i, batch_inputs in enumerate(train_data_gen):
                loss, dict_f1_score = forward_step(batch_inputs, BertCrfmodel, optimizer=optimizer)
                train_loss_metric(loss)
                [train_f1_metric[i](dict_f1_score[unit]) for i, unit in enumerate(dict_f1_score)]

                progress = {
                    "epoch": epoch + 1,
                    "progress": (i + 1) * batch_size if (i + 1) * batch_size < len(train_data) else len(train_data),
                    "BCE_loss": train_loss_metric.result().numpy()}
                f1_score = {v: k.result().numpy() for k, v in zip(train_f1_metric, list(dict_f1_score.keys()))}
                print("epoch", progress["epoch"],
                      f"{progress['progress']}/{len(train_data)}",
                      "BCE_loss", progress["BCE_loss"],
                      f"{' '.join(f1_score.keys())}", ",".join([str(k) for k in f1_score.values()])
                      )
            print("epoch:", epoch + 1,
                  "train:", train_loss_metric.result().numpy(),
                  "f1_score", {v: k.result().numpy() for k, v in zip(train_f1_metric, list(dict_f1_score.keys()))})
            for batch_inputs in valid_data_gen:
                val_loss, dict_val_f1_score = BertCrfmodel(batch_inputs)
                valid_loss_metric(val_loss)
                [valid_f1_metric[i](dict_val_f1_score[unit]) for i, unit in enumerate(dict_val_f1_score)]
            valid_f1 = {v: k.result().numpy() for k, v in zip(valid_f1_metric, list(dict_val_f1_score.keys()))}
            if np.array(valid_f1.values()).mean() >= best_f1_score:
                best_f1_score = np.array(valid_f1.values()).mean()
                BertCrfmodel.save_weights("/best_model/best_model.weights")
            print("val:", valid_loss_metric.result().numpy(),
                  "valid_f1_score",
                  {v: k.result().numpy() for k, v in zip(valid_f1_metric, list(dict_val_f1_score.keys()))},
                  "best_f1_score:", best_f1_score)
            train_loss_metric.reset_states()
            [train_f1_metric[i].reset_states() for unit in train_f1_metric]
            valid_loss_metric.reset_states()
            [valid_f1_metric[i].reset_states() for unit in valid_f1_metric]

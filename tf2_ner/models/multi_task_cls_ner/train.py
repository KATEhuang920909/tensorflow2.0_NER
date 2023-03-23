# -*- coding: utf-8 -*-
# @Time    : 2023/3/23 15:15
# @Author  : huangkai
# @File    : train.py
import tensorflow as tf
import sys
import pandas as pd

sys.path.append("../../")
from model import ClsNerModel, forward_step
from config import *
from keras.utils.vis_utils import plot_model
from bert4keras.snippets import sequence_padding
from bert4keras.tokenizers import Tokenizer
import numpy as np
from utils.data_helper import  load_dataset

categories = set()
labels = set()
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def load_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...,label2]，
              意味着text[start:end + 1]是类型为label的实体。
    """

    # ner label
    D = []
    data = pd.read_csv(filename,encoding="gbk").sample(frac=1.0)
    texts = data.text.values
    BIO_anno = data.BIO_anno.values
    label = tf.keras.utils.to_categorical(data["class"], num_classes=3)
    for i, (text, BIO) in enumerate(zip(texts, BIO_anno)):
        d = ['']
        flag = BIO.split(" ")
        d[0] = text
        for i, f in enumerate(flag):
            types = f.split("-")[0]
            if types == 'B':
                d.append([i, i, f[2:]])
                categories.add(f[2:])
            elif types == 'I':
                d[-1][1] = i
            elif types == "O":
                categories.add("others")
        d.append(label[i])
        labels.add(str(label[i]))
        D.append(d)

    return D


def tokenizers(data):
    batch_token_ids, batch_segment_ids, ner_batch_labels, cls_batch_labels = [], [], [], []
    for d in data:
        tokens = tokenizer.tokenize(d[0], maxlen=maxlen)

        mapping = tokenizer.rematch(d[0], tokens)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        ner_labels = np.zeros(len(token_ids))
        for start, end, label in d[1:-1]:
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]
                end = end_mapping[end]
                ner_labels[start] = categories.index(label) * 2 + 1
                ner_labels[start + 1:end + 1] = categories.index(label) * 2 + 2
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        ner_batch_labels.append(ner_labels)
        cls_batch_labels.append(d[-1])

    batch_token_ids = sequence_padding(batch_token_ids)
    batch_segment_ids = sequence_padding(batch_segment_ids)
    ner_batch_labels = sequence_padding(ner_batch_labels)
    return {"token_id": batch_token_ids,
            "segment_id": batch_segment_ids,
            "ner_label": ner_batch_labels,
            "cls_label": cls_batch_labels}


if __name__ == '__main__':

    data = load_data('../../data/train_data_public.csv')
    train_data, valid_data = data[:int(0.9 * (len(data)))], data[int(0.9 * (len(data))):]
    print(len(train_data), len(valid_data))
    # test_data = load_data('../../data/address/final_test.txt', is_test=True)
    categories = list(sorted(categories))
    train_data_token = tokenizers(train_data[:200])
    valid_data_token = tokenizers(valid_data)
    # print(train_data["token_id"].shape)
    train_data_gen, valid_data_gen = load_dataset(train_data_token, valid_data_token, batch_size=batch_size)

    # train_data = data_generator(train_data, batch_size=batch_size)
    clsnermodel = ClsNerModel(num_classes=len(categories))
    clsnermodel.build(input_shape={"token_id": [batch_size, maxlen],
                                    "segment_id": [batch_size, maxlen],
                                    "ner_label": [batch_size, maxlen],
                                    "cls_label": [batch_size, len(labels)]})
    print(clsnermodel.summary())
    # exit()
    plot_model(clsnermodel, to_file='BERT_multi_task_cls_ner.png', show_shapes=True)
    optimizer = tf.optimizers.Adam(learning_rate=1e-5)
    train_loss_metric = tf.keras.metrics.Mean()
    train_ner_f1_metric = tf.keras.metrics.Mean()
    train_cls_f1_metric = tf.keras.metrics.Mean()
    valid_loss_metric = tf.keras.metrics.Mean()
    valid_ner_f1_metric = tf.keras.metrics.Mean()
    valid_cls_f1_metric = tf.keras.metrics.Mean()
    # print(len(train_data))
    # exit()
    with tf.device("CPU: 0"):
        best_ner_f1_score, best_cls_f1_score = 0.0, 0.0
        for epoch in range(10):
            for i, batch_inputs in enumerate(train_data_gen):
                loss, ner_f1_score, cls_f1_score = forward_step(batch_inputs, clsnermodel, optimizer=optimizer)
                train_loss_metric(loss)
                train_ner_f1_metric(ner_f1_score)
                train_cls_f1_metric(cls_f1_score)
                progress = {
                    "epoch": epoch + 1,
                    "progress": (i + 1) * batch_size if (i + 1) * batch_size < len(train_data) else len(train_data),
                    "BCE_loss": train_loss_metric.result().numpy(),
                    "ner_f1_score": train_ner_f1_metric.result().numpy(),
                    "cls_f1_score": train_cls_f1_metric.result().numpy()}
                print("epoch", progress["epoch"],
                      f"{progress['progress']}/{len(train_data)}",
                      "BCE_loss", progress["BCE_loss"],
                      "ner_f1_score", progress["ner_f1_score"],
                      "cls_f1_score", progress["cls_f1_score"])
            print("epoch:", epoch + 1,
                  "train:", train_loss_metric.result().numpy(),
                  "ner_f1_score", train_ner_f1_metric.result().numpy(),
                  "cls_f1_score", train_cls_f1_metric.result().numpy())
            for batch_inputs in valid_data_gen:
                val_loss, val_ner_f1_score, val_cls_f1_score = clsnermodel(batch_inputs)
                valid_loss_metric(val_loss)
                valid_ner_f1_metric(val_ner_f1_score)
                valid_cls_f1_metric(val_cls_f1_score)
            valid_ner_f1, valid_cls_f1 = valid_ner_f1_metric.result().numpy(), valid_cls_f1_metric.result().numpy()
            if valid_ner_f1 >= best_ner_f1_score and valid_cls_f1 >= best_cls_f1_score:
                best_ner_f1_score, best_cls_f1_score = valid_ner_f1, valid_cls_f1
                clsnermodel.save_weights("/best_model/best_model.weights")
            print("val:", valid_loss_metric.result().numpy(),
                  "ner_f1_score", valid_ner_f1_metric.result().numpy(),
                  "cls_f1_score", valid_cls_f1_metric.result().numpy(),
                  "best_ner_f1_score:", best_ner_f1_score,
                  "best_cls_f1_score:", best_cls_f1_score)

            train_loss_metric.reset_states()
            train_ner_f1_metric.reset_states()
            train_cls_f1_metric.reset_states()
            valid_loss_metric.reset_states()
            valid_ner_f1_metric.reset_states()
            valid_cls_f1_metric.reset_states()

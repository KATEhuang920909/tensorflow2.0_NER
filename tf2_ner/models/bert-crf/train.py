# -*- coding: utf-8 -*-
# @Time    : 2023/3/12 20:08
# @Author  : huangkai
# @File    : train.py
import tensorflow as tf
import sys

sys.path.append("../../")
from model import BERTCRF2Model, forward_step
from config import *
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
    for d in data:
        tokens = tokenizer.tokenize(d[0], maxlen=maxlen)

        mapping = tokenizer.rematch(d[0], tokens)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        labels = np.zeros(len(token_ids))
        for start, end, label in d[1:]:
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]
                end = end_mapping[end]
                labels[start] = categories.index(label) * 2 + 1
                labels[start + 1:end + 1] = categories.index(label) * 2 + 2
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_labels.append(labels)

    batch_token_ids = sequence_padding(batch_token_ids, length=maxlen)
    batch_segment_ids = sequence_padding(batch_segment_ids, length=maxlen)
    batch_labels = sequence_padding(batch_labels, length=maxlen)
    return {"token_id": batch_token_ids,
            "segment_id": batch_segment_ids,
            "label": batch_labels}


class Trainer(tf.keras.Model):

    def __init__(self, emb_model):
        super(Trainer, self).__init__()
        self.emb_model = emb_model
        self.eps = 1e-9
        self.num_classes = self.emb_model.num_classes

    def call(self, inputs):
        logits = self.emb_model([inputs["token_id"], inputs["segment_id"]])
        loss = self.emb_model.CRF.sparse_loss(inputs["label"], logits)
        # f1_score = self.metric.f1_marco(logits, inputs["label"])

        # y_true = tf.reshape(inputs["label"], shapes[:-1])
        y_pred = tf.cast(tf.argmax(logits, 2), "int32")
        f1_marco, p_marco, r_marco = 0, 0, 0
        shapes = tf.shape(inputs["label"])
        ones_, zeros_ = tf.ones(shapes), tf.zeros(shapes)

        for i in range(self.num_classes * 2 + 1):
            tp = tf.reduce_sum(tf.keras.backend.switch((y_pred == i) & (inputs["label"] == i), ones_, zeros_))
            fp = tf.reduce_sum(tf.keras.backend.switch((y_pred == i) & (inputs["label"] != i), ones_, zeros_))
            fn = tf.reduce_sum(tf.keras.backend.switch((y_pred != i) & (inputs["label"] == i), ones_, zeros_))
            p = tp / (tp + fp + 1e-7)
            r = tp / (tp + fn + 1e-7)
            p_marco += p
            r_marco += r
        p_marco = p_marco / (self.num_classes * 2 + 1)
        r_marco = r_marco / (self.num_classes * 2 + 1)
        f1 = 2 * p_marco * r_marco / (p_marco + r_marco + 1e-7)
        return loss, f1


# 标注数据
if __name__ == '__main__':

    train_data = load_data('../../data/address/train.conll')
    valid_data = load_data('../../data/address/dev.conll')
    print(len(train_data), len(valid_data))
    # test_data = load_data('../../data/address/final_test.txt', is_test=True)
    categories = list(sorted(categories))
    print("categories", categories)
    train_data_token = ner_tokenizers(train_data[:200])
    valid_data_token = ner_tokenizers(valid_data[:200])
    # print(train_data["token_id"].shape)
    train_data_gen, valid_data_gen = load_dataset(train_data_token, valid_data_token, batch_size=batch_size)

    # train_data = data_generator(train_data, batch_size=batch_size)
    model = BERTCRF2Model(num_classes=len(categories))
    model.build(input_shape=[[batch_size, maxlen], [batch_size, maxlen]])

    model.compute_output_shape(input_shape=[[batch_size, maxlen], [batch_size, maxlen]])
    # tf.keras.models.save_model(model, "./", save_format="tf")
    print(model.summary())
    trainer = Trainer(model)
    # exit()
    # plot_model(model, to_file='BERT_BILSTM_CRF.png', show_shapes=True)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    train_loss_metric = tf.keras.metrics.Mean()
    train_f1_metric = tf.keras.metrics.Mean()
    valid_loss_metric = tf.keras.metrics.Mean()
    valid_f1_metric = tf.keras.metrics.Mean()
    # print(len(train_data))
    # exit()
    best_f1_score = 0.0
    for epoch in range(10):
        for i, batch_inputs in enumerate(train_data_gen):
            loss, f1_score = forward_step(batch_inputs, trainer, optimizer=optimizer,
                                          is_training=True, is_evaluate=True)
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
        for batch_inputs in valid_data_gen:
            val_loss, val_f1 = forward_step(batch_inputs, trainer, optimizer=optimizer,
                                            is_training=False, is_evaluate=True)
            valid_loss_metric(val_loss)
            valid_f1_metric(val_f1)
        valid_f1 = valid_f1_metric.result().numpy()
        if valid_f1 >= best_f1_score:
            print(f"best model update from {best_f1_score} to {valid_f1}")
            best_f1_score = valid_f1

            model.save_weights("./best_model/best_model.weights")
        print("val:", valid_loss_metric.result().numpy(),
              "val_f1_score:", valid_f1,
              "best_f1_score:", best_f1_score)
        train_loss_metric.reset_states()
        train_f1_metric.reset_states()
        valid_loss_metric.reset_states()
        valid_f1_metric.reset_states()

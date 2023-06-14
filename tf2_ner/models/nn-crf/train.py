# -*- coding: utf-8 -*-
# @Time    : 2023/3/12 20:08
# @Author  : huangkai
# @File    : train.py
import tensorflow as tf
import sys

sys.path.append("../../")
from model import NN2Model, forward_step
from utils.data_helper import get_tag2index
from tqdm.notebook import tqdm
from config import *
from keras.utils.vis_utils import plot_model
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer
import numpy as np
from utils.data_helper import DataProcess, load_dataset
from bert4keras.layers import  ConditionalRandomField,Model
categories = set()
# 建立分词器
tokenizer = Tokenizer(dict_path)


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


def get_id(data):
    """加载数据
        单条格式：[text, (start, end, label), (start, end, label), ...]，
                  意味着text[start:end + 1]是类型为label的实体。
    """
    vocab_bag = ""
    for unit in data:
        vocab_bag += unit[0]
    token_id = {}
    for chr in vocab_bag:
        if chr not in token_id:
            token_id[chr] = len(token_id)

    return len(token_id), token_id


def ner_tokenizers(data, token_id, maxlen):
    batch_token_ids, batch_labels = [], []

    for d in data:
        token_ids = []
        if len(d[0]) > maxlen:
            text = d[0][:maxlen]
        else:
            text = d[0]
        token = list(text)
        for str in text:
            token_ids.append(token_id[str])
        if len(text) < maxlen:
            token_ids += [0] * (maxlen - len(token_ids))
        mapping = tokenizer.rematch(text, token)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        labels = np.zeros(len(token_ids))
        for start, end, label in d[1:]:
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]
                end = end_mapping[end]
                labels[start] = categories.index(label) * 2 + 1
                labels[start + 1:end + 1] = categories.index(label) * 2 + 2
        batch_token_ids.append(token_ids)
        batch_labels.append(labels)
    batch_token_ids = sequence_padding(batch_token_ids)
    batch_labels = sequence_padding(batch_labels)
    return {"token_id": batch_token_ids,
            "label": batch_labels}


class Trainer(tf.keras.Model):

    def __init__(self, emb_model):
        super(Trainer, self).__init__()
        self.emb_model = emb_model
        self.eps = 1e-9
        self.num_classes = self.emb_model.num_classes
    def call(self, inputs):
        logits = self.emb_model(inputs["token_id"])
        loss = self.emb_model.CRF.sparse_loss(inputs["label"], logits)
        # f1_score = self.metric.f1_marco(logits, inputs["label"])

        # y_true = tf.reshape(inputs["label"], shapes[:-1])
        y_pred = tf.cast(tf.argmax(logits, 2), "int32")
        f1_marco = 0
        shapes = tf.shape(inputs["label"])
        ones_, zeros_ = tf.ones(shapes), tf.zeros(shapes)

        for i in range(self.num_classes * 2 + 1):
            tp = tf.reduce_sum(tf.keras.backend.switch((y_pred == i) & (inputs["label"] == i), ones_, zeros_))
            fp = tf.reduce_sum(tf.keras.backend.switch((y_pred == i) & (inputs["label"] != i), ones_, zeros_))
            fn = tf.reduce_sum(tf.keras.backend.switch((y_pred != i) & (inputs["label"] == i), ones_, zeros_))
            p = tp / (tp + fp + 1e-7)
            r = tp / (tp + fn + 1e-7)
            f1 = 2 * p * r / (p + r + 1e-7)
            f1_marco += f1

        return loss, f1_marco / (self.num_classes * 2 + 1)


# 标注数据
if __name__ == '__main__':

    train_data = load_data('../../data/address/train.conll')
    valid_data = load_data('../../data/address/dev.conll')
    print(len(train_data), len(valid_data))
    vocab_size, token_ids = get_id(train_data + valid_data)
    print(vocab_size)
    # test_data = load_data('../../data/address/final_test.txt', is_test=True)
    categories = list(sorted(categories))
    train_data_token = ner_tokenizers(train_data, token_ids, maxlen)
    valid_data_token = ner_tokenizers(valid_data, token_ids, maxlen)
    # print(train_data["token_id"].shape)
    train_data_gen, valid_data_gen = load_dataset(train_data_token, valid_data_token, batch_size=batch_size)

    # train_data = data_generator(train_data, batch_size=batch_size)
    model = NN2Model(num_classes=len(categories), vocab_size=vocab_size,
                     embed_size=embed_size, units=embed_size,
                     model_type=None, attention_type="self-attention")

    model.build(input_shape= [batch_size, maxlen])
    model.compute_output_shape(input_shape=[batch_size, maxlen])

    trainer = Trainer(model)
    # num_classes, vocab_size, embed_size, units, attention_type=None

    print(model.summary())
    # exit()
    plot_model(model, to_file='BERT_BILSTM_CRF.png', show_shapes=True)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
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
                # print(batch_inputs)
                loss, f1_score = forward_step(batch_inputs, trainer, optimizer=optimizer)
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
                val_loss, val_f1 = trainer(batch_inputs)
                valid_loss_metric(val_loss)
                valid_f1_metric(val_f1)
            valid_f1 = valid_f1_metric.result().numpy()
            if valid_f1 >= best_f1_score:
                best_f1_score = valid_f1
                trainer.save_weights("/best_model/best_model.weights")
            print("val:", valid_loss_metric.result().numpy(), valid_f1_metric.result().numpy(),
                  "best_f1_score:", best_f1_score)
            train_loss_metric.reset_states()
            train_f1_metric.reset_states()
            valid_loss_metric.reset_states()
            valid_f1_metric.reset_states()

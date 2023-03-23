#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/23 上午12:36
@Auth ： huangkai
@File ：model.py
@model_name ：biaffine
"""
import tensorflow as tf
import sys

sys.path.append("../../")
from bert4keras.models import build_transformer_model, Model
from bert4keras.layers import Dense, ConditionalRandomField
from config import *
from utils.metrics import METRICS


# define model
class BiaffineModel(tf.keras.Model):

    def __init__(self, num_classes):
        super(BiaffineModel, self).__init__()
        self.num_classes = num_classes
        output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
        bert_model = build_transformer_model(config_path, checkpoint_path)
        self.bert_model = Model(bert_model.input, bert_model.get_layer(output_layer).output, name="BERT-MODEL")
        self.dense_left = Dense(units=units, activation="relu")
        self.dense_right = Dense(units=units, activation="relu")
        self.CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
        self.metric = METRICS(num_class=self.num_classes * 2 + 1)

    def build(self, input_shape):
        self.ffns_weights = self.add_weight(name="biaffine_matrix",
                                            shape=(
                                                self.bert_model.hidden_size,
                                                self.bert_model.hidden_size // 2),
                                            initializer="uniform",
                                            trainable=True)
        self.ffne_weights = self.add_weight(name="biaffine_matrix",
                                            shape=(
                                                self.bert_model.hidden_size,
                                                self.bert_model.hidden_size // 2),
                                            initializer="uniform",
                                            trainable=True)
        self.biaffine_weights = self.add_weight(name="biaffine_matrix",
                                                shape=(
                                                    self.bert_model.hidden_size // 2,
                                                    2 * self.num_classes + 1,
                                                    self.bert_model.hidden_size // 2),
                                                initializer="uniform",
                                                trainable=True)
    def losses(self):
        """
        :todo  2023.3.22

        """
    def call(self, inputs):
        # print(inputs["token_id"].shape, inputs["label"].shape)
        outputs = self.bert_model([inputs["token_id"], inputs["segment_id"]])  # batch_size,seq_len,hidden_size

        start = tf.tanh(tf.matmul(outputs, self.ffns_weights))  # batch_size,seq_len,hidden_size//2
        end = tf.tanh(tf.matmul(outputs, self.ffne_weights))  # batch_size,seq_len,hidden_size//2
        end = tf.transpose(end, [0, 2, 1])  # batch_size,hidden_size//2,seq_len
        start = tf.reshape(start, [-1, tf.shape(start)[-1]])  # batch_size*seq_len,hidden_size//2
        self.biaffine_weights = tf.reshape(self.biaffine_weights,
                                           [tf.shape(start)[-1], -1])  # hidden_size//2,label*hidden_size//2
        step1 = tf.matmul(start, self.biaffine_weights)  # batch_size*seq_len,label*hidden_size//2
        step1 = tf.reshape(step1, [batch_size, -1, tf.shape(end)[-1]])  # batch_size,seq_len*label,hidden_size//2
        step2 = tf.matmul(step1, end)  # batch_size,seq_len*label,,seq_len
        step2 = tf.reshape(step2, [batch_size, maxlen, -1, maxlen])  # 最终的评分矩阵
        logits = tf.argmax(step2, 2)  # [batch_size, maxlen, maxlen]
        loss = self.CRF.sparse_loss(inputs["label"], logits)
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


# @tf.function(experimental_relax_shapes=True)
def forward_step(batch_inputs, trainer, optimizer=None, is_training=True):
    if is_training:
        with tf.GradientTape() as tape:
            loss, f1_score = trainer(batch_inputs)
        gradients = tape.gradient(loss, trainer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainer.trainable_variables))
    else:
        loss, f1_score = trainer(batch_inputs)
    return loss, f1_score

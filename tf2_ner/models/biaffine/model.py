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
from metrics import batch_computeF1

# define model
class BiaffineModel(tf.keras.Model):

    def __init__(self, num_classes, mask=False):
        super(BiaffineModel, self).__init__()
        self.num_classes = num_classes
        # output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
        self.bert_model = build_transformer_model(config_path, checkpoint_path)
        # self.bert_model = Model(bert_model.input, bert_model.get_layer(output_layer).output, name="BERT-MODEL")
        # print(self.bert_model.output.shape)
        self.dense_left = Dense(units=units, activation="relu")
        self.dense_right = Dense(units=units, activation="relu")
        self.CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
        self.mask = mask

    def build(self, input_shape):
        self.ffns_weights = self.add_weight(name="biaffine_matrix",
                                            shape=(
                                                self.bert_model.output.shape[-1],
                                                self.bert_model.output.shape[-1] // 2),
                                            initializer="uniform",
                                            trainable=True)
        self.ffne_weights = self.add_weight(name="biaffine_matrix",
                                            shape=(
                                                self.bert_model.output.shape[-1],
                                                self.bert_model.output.shape[-1] // 2),
                                            initializer="uniform",
                                            trainable=True)
        self.biaffine_weights = self.add_weight(name="biaffine_matrix",
                                                shape=(
                                                    self.bert_model.output.shape[-1] // 2,
                                                    self.num_classes,
                                                    self.bert_model.output.shape[-1] // 2),
                                                initializer="uniform",
                                                trainable=True)

    def call(self, inputs):
        # print(inputs["token_id"].shape, inputs["label"].shape)
        outputs = self.bert_model([inputs["token_id"], inputs["segment_id"]])  # batch_size,seq_len,hidden_size

        start = tf.nn.gelu(tf.matmul(outputs, self.ffns_weights))  # batch_size,seq_len,hidden_size//2
        end = tf.nn.gelu(tf.matmul(outputs, self.ffne_weights))  # batch_size,seq_len,hidden_size//2
        # end = tf.transpose(end, [0, 2, 1])  # batch_size,hidden_size//2,seq_len
        # start = tf.reshape(start, [-1, tf.shape(start)[-1]])  # batch_size*seq_len,hidden_size//2
        # self.biaffine_weights = tf.reshape(self.biaffine_weights,
        #                                    [tf.shape(start)[-1], -1])  # hidden_size//2,label*hidden_size//2
        # step1 = tf.matmul(start, self.biaffine_weights)  # batch_size*seq_len,label*hidden_size//2
        # step1 = tf.reshape(step1, [batch_size, -1, tf.shape(end)[1]])  # batch_size,seq_len*label,hidden_size//2
        # print(step1.shape, end.shape)
        # step2 = tf.einsum("ijk,ikn->ijn", step1, end)  # batch_size,seq_len*label,,seq_len
        # print(step2.shape,batch_size, maxlen, maxlen, self.num_classes)
        # logits = tf.reshape(step2, [batch_size, maxlen, maxlen, self.num_classes])  # 最终的评分矩阵
        # print(logits.shape,inputs["label"].shape)
        logits = tf.einsum("bxi,ioj,byj->bxyo", start, self.biaffine_weights, end)
        # s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        # logits = tf.squeeze(logits,1)
        logits = tf.nn.softmax(logits)
        if self.mask:
            mask = tf.ones(shape=[batch_size, maxlen, maxlen])
            mask = tf.linalg.LinearOperatorLowerTriangular(mask).to_dense()
            mask = tf.expand_dims(mask, -1)
            mask = tf.concat([mask] * (self.num_classes+1), axis=-1)
            logits = tf.multiply(logits, mask)  # 下三角
            y_true = tf.multiply(inputs["label"], mask) # 下三角
            # print(logits.shape,logits)
            loss = tf.nn.softmax_cross_entropy_with_logits(y_true, logits)
            loss = tf.reduce_mean(loss)
        else:
            loss = tf.nn.softmax_cross_entropy_with_logits(inputs["label"], logits)
            loss = tf.reduce_mean(loss)
        y_pred = tf.cast(tf.argmax(logits, -1), "int32")
        y_true = tf.cast(tf.argmax(inputs["label"], -1), "int32")
        f1_marco = []
        # shapes = tf.shape(y_true)
        # ones_, zeros_ = tf.ones(shapes), tf.zeros(shapes)
        #
        # for i in range(self.num_classes + 1):
        #     tp = tf.reduce_sum(tf.keras.backend.switch((y_pred == i) & (y_true == i), ones_, zeros_))
        #     fp = tf.reduce_sum(tf.keras.backend.switch((y_pred == i) & (y_true != i), ones_, zeros_))
        #     fn = tf.reduce_sum(tf.keras.backend.switch((y_pred != i) & (y_true == i), ones_, zeros_))
        #     p = tp / (tp + fp + 1e-7)
        #     r = tp / (tp + fn + 1e-7)
        #     f1 = 2 * p * r / (p + r + 1e-7)
        #     f1_marco.append(f1)
        precision, recall, f1_score, report = batch_computeF1(y_true, y_pred, maxlen, self.num_classes+1)

        return loss, f1_score


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

#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/26 上午12:35
@Auth ： huangkai
@File ：cascade_span.py
@IDE ：PyCharm
"""
import tensorflow as tf
import sys

sys.path.append("../../")
from bert4keras.models import build_transformer_model, Model
from bert4keras.layers import Dense, ConditionalRandomField
from config import *


# define model
class CascadeSpan(tf.keras.Model):

    def __init__(self, num_classes, mask=False):
        super(CascadeSpan, self).__init__()
        self.num_classes = num_classes
        self.bert_model = build_transformer_model(config_path, checkpoint_path)
        self.dense_head_stage1 = Dense(units=units, activation="sigmoid")
        self.dense_tail_stage1 = Dense(units=units, activation="sigmoid")
        self.dense_head_stage12 = Dense(units=1, activation="sigmoid")
        self.dense_tail_stage12 = Dense(units=1, activation="sigmoid")

        self.dense_head_stage2 = Dense(units=units, activation="sigmoid")
        self.dense_tail_stage2 = Dense(units=units, activation="sigmoid")
        self.dense_head_stage22 = Dense(units=1, activation="sigmoid")
        self.dense_tail_stage22 = Dense(units=1, activation="sigmoid")
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
        """
                : todo  here 20230327
                """
        # print(inputs["token_id"].shape, inputs["label"].shape)
        outputs = self.bert_model([inputs["token_id"], inputs["segment_id"]])  # batch_size,seq_len,hidden_size
        # 第一阶段
        head_outputs = self.dense_head_stage1(outputs)
        tail_outputs = self.dense_tail_stage1(outputs)
        head_pred = self.dense_head_stage12(head_outputs)
        tail_pred = self.dense_tail_stage12(tail_outputs)

        head_bio_label = inputs["head_bio_labels"]
        tail_bio_label = inputs["tail_bio_labels"]

        head_loss = tf.losses.binary_crossentropy(head_bio_label, head_pred)
        tail_loss = tf.losses.binary_crossentropy(tail_bio_label, tail_pred)

        # 第二阶段

        if self.mask:
            head_bio_slice = tf.multiply(tf.range(maxlen), tf.round(head_pred))
            tail_bio_slice = tf.multiply(tf.range(maxlen), tf.round(tail_pred))
            head_cls = tf.gather_nd(head_outputs, head_bio_slice)
            tail_cls = tf.gather_nd(tail_outputs, tail_bio_slice)
            head_outputs2 = self.dense_head_stage2(head_cls)
            tail_outputs2 = self.dense_tail_stage2(tail_cls)
            head_pred2 = self.dense_head_stage22(head_outputs2)
            tail_pred2 = self.dense_tail_stage22(tail_outputs2)

            head_cls_labels = tf.gather_nd(inputs["head_cls_labels"], head_bio_slice)
            tail_cls_labels = tf.gather_nd(inputs["tail_cls_labels"], tail_bio_slice)
            head_loss2 = tf.losses.binary_crossentropy(head_cls_labels, head_pred2)
            tail_loss2 = tf.losses.binary_crossentropy(tail_cls_labels, tail_pred2)
        else:
            head_outputs2 = self.dense_head_stage2(head_outputs)
            tail_outputs2 = self.dense_tail_stage2(tail_outputs)
            head_pred2 = self.dense_head_stage22(head_outputs2)
            tail_pred2 = self.dense_tail_stage22(tail_outputs2)
            head_cls_labels, tail_cls_labels = inputs["head_cls_labels"], inputs["tail_cls_labels"]
            head_loss2 = tf.losses.binary_crossentropy(head_cls_labels, head_pred2)
            tail_loss2 = tf.losses.binary_crossentropy(tail_cls_labels, tail_pred2)

        loss = (head_loss + tail_loss) + (head_loss2 + tail_loss2)

        f1_marco_head_bio, f1_marco_tail_bio, f1_marco_head_cls, f1_marco_tail_cls = 0, 0, 0, 0
        shapes = tf.shape(head_bio_label), tf.shape(tail_bio_label), tf.shape(head_cls_labels), tf.shape(
            tail_cls_labels)
        ones_, zeros_ = tf.ones(shapes), tf.zeros(shapes)

        for i in range(2):
            tp = tf.reduce_sum(tf.keras.backend.switch((y_pred == i) & (y_true == i), ones_, zeros_))
            fp = tf.reduce_sum(tf.keras.backend.switch((y_pred == i) & (y_true != i), ones_, zeros_))
            fn = tf.reduce_sum(tf.keras.backend.switch((y_pred != i) & (y_true == i), ones_, zeros_))
            p = tp / (tp + fp + 1e-7)
            r = tp / (tp + fn + 1e-7)
            f1 = 2 * p * r / (p + r + 1e-7)
            f1_marco += f1

        return loss, f1_marco / (self.num_classes)


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

# -*- coding: utf-8 -*-
# @Time    : 2023/3/7 10:45
# @Author  : huangkai
# @File    : model.py
import gc

import numpy as np
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from tqdm.notebook import tqdm
import sys

sys.path.append("../../")
from bert4keras.models import build_transformer_model
from bert4keras.layers import Dense, ConditionalRandomField
from bert4keras.snippets import ViterbiDecoder
from config import *
from utils.metrics import METRICS


# define model
class BERTCRF2Model(tf.keras.Model):

    def __init__(self, feat_dim, out_dim, num_linear):
        super(BERTCRF2Model, self).__init__()
        self.bert_model = build_transformer_model(config_path, checkpoint_path)
        self.CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)

    def call(self, inputs):
        output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
        outputs = self.bert_model.get_layer(output_layer).output
        outputs = Dense(len(categories) * 2 + 1)(outputs)

        output = self.CRF(outputs)  # logits
        return output


class Predictor(tf.keras.Model):

    def __init__(self, out_dim):
        super(Predictor, self).__init__()
        self.out_layer = tf.keras.layers.Dense(out_dim)

    def call(self, inputs):
        return self.out_layer(inputs)


class Trainer(tf.keras.Model, ViterbiDecoder):

    def __init__(self, model, num_class):
        super(Trainer, self).__init__()
        self.model = model
        self.eps = 1e-9
        self.metric = METRICS(num_class=num_class)

    def call(self, inputs):
        """
        # 计算loss、acc、f1值
        """
        logits = self.model(inputs[0])  # logits
        labels = inputs[1]
        loss_sum = 0.
        pos_true_positive = 0.
        pos_false_positive = 0.
        pos_false_negative = 0.
        neg_true_positive = 0.
        neg_false_positive = 0.
        neg_false_negative = 0.
        correct = 0.
        for i in range(3):
            pred_emb = tf.reduce_sum(tf.gather(unique_emb, inputs[f"history_{i}"]), axis=1)
            pred_val = tf.clip_by_value(tf.math.sigmoid(self.predictors[i](tf.nn.l2_normalize(pred_emb, axis=1))),
                                        self.eps, 1. - self.eps)
            # Binary Cross Entropy
            loss = -inputs[f"label_{i}"] * tf.math.log(pred_val) - (1. - inputs[f"label_{i}"]) * tf.math.log(
                1. - pred_val)
            loss_sum += tf.reduce_sum(tf.reduce_mean(loss, axis=0))

            # F1-macro
            pred_label = pred_val > .5
            bool_label = inputs[f"label_{i}"] == 1.
            correct += tf.cast(tf.math.count_nonzero(pred_label == bool_label), "float32")
            pos_true_positive += tf.cast(tf.math.count_nonzero(tf.math.logical_and(bool_label, pred_label)), "float32")
            pos_false_positive += tf.cast(
                tf.math.count_nonzero(tf.math.logical_and(tf.math.logical_not(bool_label), pred_label)), "float32")
            pos_false_negative += tf.cast(
                tf.math.count_nonzero(tf.math.logical_and(bool_label, tf.math.logical_not(pred_label))), "float32")

            pred_label = pred_val < .5
            bool_label = inputs[f"label_{i}"] == 0.
            neg_true_positive += tf.cast(tf.math.count_nonzero(tf.math.logical_and(bool_label, pred_label)), "float32")
            neg_false_positive += tf.cast(
                tf.math.count_nonzero(tf.math.logical_and(tf.math.logical_not(bool_label), pred_label)), "float32")
            neg_false_negative += tf.cast(
                tf.math.count_nonzero(tf.math.logical_and(bool_label, tf.math.logical_not(pred_label))), "float32")

        accuracy = correct / tf.cast(tf.shape(inputs[f"label_0"])[0] * 18, "float32")
        pos_recall = pos_true_positive / tf.maximum(self.eps, pos_true_positive + pos_false_negative)
        pos_precision = pos_true_positive / tf.maximum(self.eps, pos_true_positive + pos_false_positive)
        pos_f1 = 2 * pos_recall * pos_precision / tf.maximum(self.eps, pos_recall + pos_precision)

        neg_recall = neg_true_positive / tf.maximum(self.eps, neg_true_positive + neg_false_negative)
        neg_precision = neg_true_positive / tf.maximum(self.eps, neg_true_positive + neg_false_positive)
        neg_f1 = 2 * neg_recall * neg_precision / tf.maximum(self.eps, neg_recall + neg_precision)

        return loss_sum, (pos_f1 + neg_f1) / 2., accuracy

    def predict_proba(self, inputs):
        unique_emb = self.model(inputs["unique_feature"])
        labels = []
        pred_vals = []
        for i in range(3):
            pred_emb = tf.reduce_sum(tf.gather(unique_emb, inputs[f"history_{i}"]), axis=1)
            pred_val = tf.clip_by_value(tf.math.sigmoid(self.predictors[i](tf.nn.l2_normalize(pred_emb, axis=1))),
                                        self.eps, 1. - self.eps)
            pred_vals.append(pred_val)
            labels.append(inputs[f"label_{i}"])
        return tf.concat(pred_vals, axis=1), tf.concat(labels, axis=1)


@tf.function(experimental_relax_shapes=True)
def forward_step(trainer, optimizer, batch_inputs):
    with tf.GradientTape() as tape:
        loss, f1, acc = trainer(batch_inputs, training=True)
    gradients = tape.gradient(loss, trainer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainer.trainable_variables))
    return loss, f1, acc

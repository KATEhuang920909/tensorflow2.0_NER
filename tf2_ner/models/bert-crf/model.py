# -*- coding: utf-8 -*-
# @Time    : 2023/3/7 10:45
# @Author  : huangkai
# @File    : model.py
import tensorflow as tf
import sys

sys.path.append("../../")
from bert4keras.models import build_transformer_model, Model
from bert4keras.layers import Dense, ConditionalRandomField
from config import *


# define model
class BERTCRF2Model(tf.keras.Model):

    def __init__(self, num_classes):
        super(BERTCRF2Model, self).__init__()
        self.num_classes = num_classes
        output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
        bert_model = build_transformer_model(config_path, checkpoint_path)
        self.bert_model = Model(bert_model.input, bert_model.get_layer(output_layer).output, name="BERT-MODEL")
        self.dense_layer = Dense(self.num_classes * 2 + 1, activation="relu")
        self.CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)

    def call(self, inputs):
        # print(inputs["token_id"].shape, inputs["label"].shape)
        outputs = self.bert_model([inputs["token_id"], inputs["segment_id"]])

        outputs = self.dense_layer(outputs)
        logits = self.CRF(outputs)  # logits

        loss = self.CRF.sparse_loss(inputs["label"], logits)
        # f1_score = self.metric.f1_marco(logits, inputs["label"])

        # y_true = tf.reshape(inputs["label"], shapes[:-1])

        return loss, logits


def F1_Score(y_true, y_pred, num_classes):
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(tf.argmax(y_pred, 2), "int32")
    f1_marco = []
    shapes = tf.shape(y_true)
    ones_, zeros_ = tf.ones(shapes), tf.zeros(shapes)
    # print(y_pred, y_true)
    for i in range(num_classes * 2 + 1):
        tp = tf.reduce_sum(tf.keras.backend.switch((y_pred == i) & (y_true == i), ones_, zeros_))
        fp = tf.reduce_sum(tf.keras.backend.switch((y_pred == i) & (y_true != i), ones_, zeros_))
        fn = tf.reduce_sum(tf.keras.backend.switch((y_pred != i) & (y_true == i), ones_, zeros_))
        p = tp / (tp + fp + 1e-7)
        r = tp / (tp + fn + 1e-7)
        f1 = 2 * p * r / (p + r + 1e-7)
        f1_marco.append(f1)
    return tf.reduce_mean(f1_marco)


# @tf.function(experimental_relax_shapes=True)
def forward_step(batch_inputs, trainer, num_classes, optimizer=None, is_training=True, is_evaluate=True):
    if is_training:
        with tf.GradientTape() as tape:
            loss, logits = trainer(batch_inputs)

            gradients = tape.gradient(loss, trainer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainer.trainable_variables))
    else:
        loss, logits = trainer(batch_inputs)
    if is_evaluate:
        f1_score = F1_Score(batch_inputs["label"], logits, num_classes)
        return loss, f1_score
    else:
        return loss, logits

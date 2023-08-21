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
from bert4keras.backend import K


# define model
class BERTCRF2Model(tf.keras.Model):

    def __init__(self, num_classes):
        super(BERTCRF2Model, self).__init__()
        self.num_classes = num_classes
        # output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
        bert_model = build_transformer_model(config_path, checkpoint_path)
        self.bert_model = Model(bert_model.input, bert_model.output, name="BERT-MODEL")
        self.dense_layer = Dense(self.num_classes * 2 + 1, activation="relu")
        self.CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)

    def build(self, input_shape):
        super(BERTCRF2Model, self).build(input_shape)

    def call(self, inputs):
        # print(inputs)
        outputs = self.bert_model([inputs])

        outputs = self.dense_layer(outputs)
        logits = self.CRF(outputs)  # logits

        # f1_score = self.metric.f1_marco(logits, inputs["label"])

        # y_true = tf.reshape(inputs["label"], shapes[:-1])

        return logits


# @tf.function(experimental_relax_shapes=True)
def forward_step(batch_inputs, trainer, optimizer=None, is_training=True, is_evaluate=True):
    if is_training:
        with tf.GradientTape() as tape:
            loss, f1_score = trainer(batch_inputs)

            gradients = tape.gradient(loss, trainer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainer.trainable_variables))
        return loss, f1_score
    if is_evaluate:
        _, f1_score = trainer(batch_inputs)
        return _, f1_score

# -*- coding: utf-8 -*-
# @Time    : 2023/3/23 15:14
# @Author  : huangkai
# @File    : model.py
import tensorflow as tf
import sys

sys.path.append("../../")
from bert4keras.models import build_transformer_model, Model
from bert4keras.layers import Dense, ConditionalRandomField, Bidirectional, LSTM, Lambda
from config import *
from utils.metrics import METRICS


# define model
class ClsNerModel(tf.keras.Model):

    def __init__(self, num_classes):
        super(ClsNerModel, self).__init__()
        self.num_classes = num_classes
        output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
        bert_model = build_transformer_model(config_path, checkpoint_path)
        self.ner_bert_model = Model(bert_model.input, bert_model.get_layer(output_layer).output, name="NER_BERT-MODEL")
        self.cls_bert_model = Model(bert_model.input, Lambda(lambda x: x[:, 0])(bert_model.output),
                                    name="CLS_BERT-MODEL")
        # self.cls_lstm = Bidirectional(LSTM(units=units,return_sequences=True))
        self.cls_dense_layer = Dense(units=3, activation="softmax", kernel_initializer=self.cls_bert_model.initializer)
        self.ner_dense_layer = Dense(self.num_classes * 2 + 1, activation="relu")
        self.CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
        self.metric = METRICS(num_class=self.num_classes * 2 + 1)
        self.cls_loss = tf.keras.losses.BinaryFocalCrossentropy(from_logits=True)

    def call(self, inputs):
        # print(inputs["token_id"].shape, inputs["label"].shape)
        ner_outputs = self.ner_bert_model([inputs["token_id"], inputs["segment_id"]])
        cls_outputs = self.cls_bert_model([inputs["token_id"], inputs["segment_id"]])

        ner_outputs = self.ner_dense_layer(ner_outputs)
        ner_logits = self.CRF(ner_outputs)  # logits

        cls_logits = self.cls_dense_layer(cls_outputs)

        ner_loss = self.CRF.sparse_loss(inputs["ner_label"], ner_logits)

        cls_loss = self.cls_loss(inputs["cls_label"], cls_logits)

        loss = ner_loss + cls_loss

        ner_pred = tf.cast(tf.argmax(ner_logits, -1), "int32")
        cls_pred = tf.cast(tf.argmax(cls_logits, -1), "int32")

        ner_f1_marco, cls_f1_marco = 0, 0
        ner_shapes = tf.shape(inputs["ner_label"])
        ones_, zeros_ = tf.ones(ner_shapes), tf.zeros(ner_shapes)

        for i in range(self.num_classes * 2 + 1):
            tp = tf.reduce_sum(tf.keras.backend.switch((ner_pred == i) & (inputs["ner_label"] == i), ones_, zeros_))
            fp = tf.reduce_sum(tf.keras.backend.switch((ner_pred == i) & (inputs["ner_label"] != i), ones_, zeros_))
            fn = tf.reduce_sum(tf.keras.backend.switch((ner_pred != i) & (inputs["ner_label"] == i), ones_, zeros_))
            p = tp / (tp + fp + 1e-7)
            r = tp / (tp + fn + 1e-7)
            ner_f1 = 2 * p * r / (p + r + 1e-7)
            ner_f1_marco += ner_f1

        cls_shapes = tf.shape(inputs["cls_label"])
        ones_, zeros_ = tf.ones(cls_shapes), tf.zeros(cls_shapes)

        for i in range(3):
            tp = tf.reduce_sum(tf.keras.backend.switch((cls_pred == i) & (inputs["cls_label"] == i), ones_, zeros_))
            fp = tf.reduce_sum(tf.keras.backend.switch((cls_pred == i) & (inputs["cls_label"] != i), ones_, zeros_))
            fn = tf.reduce_sum(tf.keras.backend.switch((cls_pred != i) & (inputs["cls_label"] == i), ones_, zeros_))
            p = tp / (tp + fp + 1e-7)
            r = tp / (tp + fn + 1e-7)
            cls_f1 = 2 * p * r / (p + r + 1e-7)
            cls_f1_marco += cls_f1

        return loss, ner_f1_marco / (self.num_classes * 2 + 1), cls_f1_marco / 3


# @tf.function(experimental_relax_shapes=True)
def forward_step(batch_inputs, trainer, optimizer=None, is_training=True):
    if is_training:
        with tf.GradientTape() as tape:
            loss, ner_f1_score, cls_f1_score = trainer(batch_inputs)
        gradients = tape.gradient(loss, trainer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainer.trainable_variables))
    else:
        loss, ner_f1_score, cls_f1_score = trainer(batch_inputs)
    return loss, ner_f1_score, cls_f1_score

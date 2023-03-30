# -*- coding: utf-8 -*-
# @Time    : 2023/3/7 10:45
# @Author  : huangkai
# @File    : model.py
import tensorflow as tf
import sys

sys.path.append("../")
sys.path.append("../../")
from bert4keras.layers import Dense, ConditionalRandomField, Dropout
from config import *
from utils.metrics import METRICS
from attention import SelfAttention, Attention


# define model
class NN2Model(tf.keras.Model):
    """
    todo: 完成 idcnn-crf
    """

    def __init__(self, num_classes, vocab_size, embed_size, units,
                 model_type="bilstm", attention_type=None, drop_rate: float = 0.3, ):
        super(NN2Model, self).__init__()
        self.model_type = model_type
        self.num_classes = num_classes
        self.Embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_size, mask_zero=True)
        self.LSTM = tf.keras.layers.LSTM(units=units, return_sequences=True)
        self.BILSTM = tf.keras.layers.Bidirectional(self.LSTM)
        self.dense_layer = Dense(self.num_classes * 2 + 1, activation="relu")
        self.CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
        self.dropout = Dropout(rate=drop_rate)
        self.conv = tf.keras.layers.Conv1D(filters=256,
                                           kernel_size=2,
                                           activation='relu',
                                           padding='same',
                                           dilation_rate=1)
        self.conv2 = tf.keras.layers.Conv1D(filters=256,
                                            kernel_size=3,
                                            activation='relu',
                                            padding='same',
                                            dilation_rate=1)
        self.idcnn = tf.keras.layers.Conv1D(filters=512,
                                            kernel_size=4,
                                            activation='relu',
                                            padding='same',
                                            dilation_rate=2)
        if attention_type == "attention":
            self.attention = Attention(units)
        elif attention_type == "self-attention":
            self.attention = SelfAttention(units)
        else:
            self.attention = None

    def call(self, inputs):
        # print(inputs["token_id"].shape, inputs["label"].shape)
        outputs = self.Embedding(inputs["token_id"])
        if self.model_type == "bilstm":
            outputs = self.BILSTM(outputs)
            outputs = self.Dropout(outputs)
        elif self.model_type == "idcnn":
            outputs = self.conv(outputs)
            outputs = self.conv2(outputs)
            outputs = self.idcnn(outputs)
            outputs = self.Dropout(outputs)
        if self.attention:
            outputs = self.attention(outputs)
        else:
            outputs = tf.reduce_mean(outputs, axis=-1)
        outputs = self.dense_layer(outputs)
        logits = self.CRF(outputs)  # logits

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

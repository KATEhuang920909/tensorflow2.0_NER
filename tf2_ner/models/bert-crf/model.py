# -*- coding: utf-8 -*-
# @Time    : 2023/3/7 10:45
# @Author  : huangkai
# @File    : model.py
import tensorflow as tf
from bert4keras.models import build_transformer_model, Model
from bert4keras.layers import Dense, ConditionalRandomField
from config import *
from utils.metrics import METRICS
from bert4keras.backend import K


# define model
class BERTCRF2Model(tf.keras.Model):

    def __init__(self, num_classes):
        super(BERTCRF2Model, self).__init__()
        self.num_classes = num_classes
        output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
        bert_model = build_transformer_model(config_path, checkpoint_path)
        self.bert_model = Model(bert_model.input, bert_model.get_layer(output_layer).output,name="BERT-MODEL")
        self.dense_layer = Dense(self.num_classes * 2 + 1,activation="relu")
        self.CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
        self.metric = METRICS(num_class=self.num_classes * 2 + 1)

    def call(self, inputs):
        # print(inputs["token_id"].shape, inputs["label"].shape)
        outputs = self.bert_model([inputs["token_id"], inputs["segment_id"]])

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
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 * p * r / (p + r)
            f1_marco += f1

        return loss, f1_marco / (self.num_classes * 2 + 1)


# @tf.function(experimental_relax_shapes=True)
def forward_step(batch_inputs, trainer, optimizer):
    with tf.GradientTape() as tape:
        loss, f1_score = trainer(batch_inputs, training=True)

    gradients = tape.gradient(loss, trainer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainer.trainable_variables))
    return loss, f1_score

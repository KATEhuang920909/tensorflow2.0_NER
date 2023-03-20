# -*- coding: utf-8 -*-
# @Time    : 2023/3/7 10:45
# @Author  : huangkai
# @File    : model.py
import tensorflow as tf
from bert4keras.models import build_transformer_model
from bert4keras.layers import Dense, ConditionalRandomField
from config import *
from utils.metrics import METRICS


# define model
class BERTCRF2Model(tf.keras.Model):

    def __init__(self, num_classes):
        super(BERTCRF2Model, self).__init__()
        self.num_classes = num_classes
        self.bert_model = build_transformer_model(config_path, checkpoint_path)

        self.CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
        self.metric = METRICS(num_class=len(self.num_classes) * 2 + 1)

    def call(self, inputs):
        output = self.bert_model(inputs[0])
        output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
        outputs = output.get_layer(output_layer).output
        outputs = Dense(len(self.num_classes) * 2 + 1)(outputs)
        logits = self.CRF(outputs)  # logits
        loss = self.CRF.sparse_loss(logits, inputs[1])
        f1_score = self.metric.f1_marco(logits, inputs[0])

        return loss, f1_score


# @tf.function(experimental_relax_shapes=True)
def forward_step(batch_inputs, trainer, optimizer):
    with tf.GradientTape() as tape:
        loss, f1_score = trainer(batch_inputs, training=True)

    gradients = tape.gradient(loss, trainer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainer.trainable_variables))
    return loss, f1_score

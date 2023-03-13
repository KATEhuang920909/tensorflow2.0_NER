# -*- coding: utf-8 -*-
# @Time    : 2023/3/12 20:08
# @Author  : huangkai
# @File    : train.py
import tensorflow as tf
import sys

sys.path.append("../../")
from model import BERTCRF2Model, Predictor, forward_step, Trainer
from utils.data_helper import DataProcess, load_dataset, get_tag2index
from tqdm.notebook import tqdm
from config import *
from keras.utils.vis_utils import plot_model

dp = DataProcess()
train_data, train_label, val_data, val_label = dp.get_data(one_hot=False)
print(train_data.shape, train_label.shape, val_data.shape, val_label.shape)

train_dataset, val_dataset = load_dataset(train_data, train_label, val_data, val_label, batch_size=batch_size)
# define model and dataset
# data_loader = DataLoader(train_histories, train_labels, unique_feature)
bertcrf2model = BERTCRF2Model(feature_num_vocabs, feat_dim=8, out_dim=32, num_cross=5, num_linear=0)
plot_model(bertcrf2model, to_file='BERT_BILSTM_CRF.png', show_shapes=True)

predictors = [Predictor(3), Predictor(10), Predictor(5)]
trainer = Trainer(bertcrf2model, len(get_tag2index()))
optimizer = tf.optimizers.Adam(learning_rate=1e-3)
loss_metric = tf.keras.metrics.Mean()
f1_metric = tf.keras.metrics.Mean()
acc_metric = tf.keras.metrics.Mean()

with tf.device("CPU: 0"):
    best_f1_score = 0.0
    for epoch in range(10):
        with tqdm(total=len(train_dataset)) as pbar:
            for batch_inputs in train_dataset:
                loss, f1, acc = forward_step(trainer=trainer, optimizer=optimizer, batch_inputs=batch_inputs)
                loss_metric(loss)
                f1_metric(f1)
                acc_metric(acc)
                progress = {"BCE": loss_metric.result().numpy(), "f1": f1_metric.result().numpy(),
                            "accuracy": acc_metric.result().numpy()}
                pbar.set_postfix(progress)
                pbar.update(1)
        print("epoch:", epoch + 1)
        print("train:", loss_metric.result().numpy(), f1_metric.result().numpy())
        loss_metric.reset_states()
        f1_metric.reset_states()
        acc_metric(acc)

        with tqdm(total=len(val_dataset)) as pbar:
            final_loss, final_f1, final_acc = 0.0, 0.0, 0.0
            for i, batch_inputs in enumerate(val_dataset):
                val_loss, val_f1, val_acc =trainer(batch_inputs, training=True)
                final_loss += val_loss
                final_f1 += val_f1
                final_acc += val_acc
            if final_f1 >= best_f1_score:
                best_f1_score = final_f1
                emb_model.save_weights("/best_model/best_model.weights")
        print("val:", final_loss / (i + 1), final_f1 / (i + 1), final_acc / (i + 1), "best_f1_score:", best_f1_score)

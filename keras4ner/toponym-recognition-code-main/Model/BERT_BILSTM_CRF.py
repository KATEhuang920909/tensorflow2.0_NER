"""
采用 BERT + BILSTM + CRF 网络进行处理
"""

from Public.path import path_bert_dir
from Public.metrics import METRICS
from keras.models import Model
from keras.layers import Bidirectional, LSTM, Dense, Dropout
import os
from bert4keras.optimizers import Adam
from bert4keras.models import build_transformer_model
from bert4keras.layers import ConditionalRandomField as CRF

class BERTBILSTMCRF(object):
    def __init__(self,
                 vocab_size: int,
                 n_class: int,
                 max_len: int = 100,
                 embedding_dim: int = 128,
                 rnn_units: int = 128,
                 drop_rate: float = 0.5,
                 ):
        self.vocab_size = vocab_size
        self.n_class = n_class
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.drop_rate = drop_rate
        self.config_path = os.path.join(path_bert_dir, 'bert_config.json')
        self.check_point_path = os.path.join(path_bert_dir, 'bert_model.ckpt')
        self.dict_path = os.path.join(path_bert_dir, 'vocab.txt')

    def creat_model(self):
        print('load bert Model start!')
        model = build_transformer_model(self.config_path, checkpoint_path=self.check_point_path)
        print('load bert Model end!')
        inputs = model.inputs
        embedding = model.output
        x = Bidirectional(LSTM(units=self.rnn_units, return_sequences=True))(embedding)
        x = Dropout(self.drop_rate)(x)
        x = Dense(self.n_class)(x)
        self.crf = CRF(lr_multiplier=1000)
        x = self.crf(x)
        self.model = Model(inputs=inputs, outputs=x)
        self.model.summary()
        self.compile()

        return self.model

    def compile(self):
        self.model.compile(optimizer=Adam(1e-5),
                           loss=self.crf.sparse_loss,
                           metrics=[self.crf.sparse_accuracy])


if __name__ == '__main__':
    from DataProcess.process_data import DataProcess
    from keras.utils.vis_utils import plot_model

    dp = DataProcess(data_type='renmin', model='bert')
    train_data, train_label, test_data, test_label = dp.get_data(one_hot=False)
    md = BERTBILSTMCRF(vocab_size=dp.vocab_size, n_class=dp.tag_size)
    md.creat_model()
    model = md.model

    # plot_model(model, to_file='picture/BERT_BILSTM_CRF.png', show_shapes=True)
    #
    # exit()

    model.fit(train_data, train_label, batch_size=4, epochs=2,
              validation_data=[test_data, test_label])

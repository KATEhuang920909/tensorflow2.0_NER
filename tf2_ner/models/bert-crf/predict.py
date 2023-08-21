# -*- coding: utf-8 -*-
# @Time    : 2023/3/12 20:08
# @Author  : huangkai
# @File    : predict.py


from train import *
from config import *
from bert4keras.snippets import ViterbiDecoder, to_array
from bert4keras.backend import K

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """

    def recognize(self, text):
        tokens = tokenizer.tokenize(text, maxlen=512)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], categories[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False
        return [(mapping[w[0]][0], mapping[w[-1]][-1], l) for w, l in entities]


if __name__ == '__main__':
    train_data = load_data('../../data/address/train.conll')
    valid_data = load_data('../../data/address/dev.conll')
    print(len(train_data), len(valid_data))
    # test_data = load_data('../../data/address/final_test.txt', is_test=True)
    categories = list(sorted(categories))
    model = BERTCRF2Model(len(categories))
    model.build(input_shape=[[batch_size, maxlen], [batch_size, maxlen]])

    # model.compute_output_shape(input_shape=[[batch_size, maxlen], [batch_size, maxlen]])
    print(model.summary())
    model.load_weights("./best_model/best_model.weights")
    # model.build((2,1,50))

    NER = NamedEntityRecognizer(trans=K.eval(model.get_layer("conditional_random_field").trans), starts=[0], ends=[0])
    # NER.trans = K.eval(model.CRF.trans)
    print(NER.recognize("我这次来武汉只做三件事"))

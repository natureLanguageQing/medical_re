#! -*- coding:utf-8 -*-
# 三元组抽取任务，基于GlobalPointer的仿TPLinker设计

import json
import time

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.backend import sparse_multilabel_categorical_crossentropy
from bert4keras.layers import EfficientGlobalPointer as GlobalPointer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import open, to_array
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer
from pydantic import BaseModel
from ray.serve import HTTPOptions
from starlette.middleware.cors import CORSMiddleware

maxlen = 128
config_path = '../chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = None
dict_path = '../chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：{'text': text, 'spo_list': [(s, p, o)]}
    """
    D = []
    role_name = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            text = l['text']
            event_list = l['relation']
            predicate_list = []
            object_list = []
            subject_list = []
            for o in event_list:
                subject = o['node_1']

                predicate = o['relation']
                object = o['node_2']
                predicate_list.append(predicate)
                object_list.append(object)
                subject_list.append(subject)
            D.append({
                'text': text,
                'spo_list': [(subject_list[i], predicate_list[i], object_list[i])
                             for i in range(len(predicate_list))]
            })
            role_name.extend(predicate_list)
    role_name = list(set(role_name))
    return D, role_name


# 加载数据集
all_data, role_all = load_data('../answer_triple.json')
valid_data = all_data[int(len(all_data) * 0.7):]
train_data = all_data[:int(len(all_data) * 0.7)]
predicate2id, id2predicate = {}, {}
for i in range(len(role_all)):
    if role_all[i] not in predicate2id:
        id2predicate[len(predicate2id)] = role_all[i]
        predicate2id[role_all[i]] = len(predicate2id)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)








def extract_spoes(text, model, threshold=0):
    """抽取输入text所包含的三元组
    """
    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    outputs = model.predict([token_ids, segment_ids])
    outputs = [o[0] for o in outputs]
    # 抽取subject和object
    subjects, objects = set(), set()
    outputs[0][:, [0, -1]] -= np.inf
    outputs[0][:, :, [0, -1]] -= np.inf
    for l, h, t in zip(*np.where(outputs[0] > threshold)):
        if l == 0:
            subjects.add((h, t))
        else:
            objects.add((h, t))
    # 识别对应的predicate
    spoes = set()
    for sh, st in subjects:
        for oh, ot in objects:
            p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
            p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
            ps = set(p1s) & set(p2s)
            for p in ps:
                spoes.add((
                    text[mapping[sh][0]:mapping[st][-1] + 1], id2predicate[p],
                    text[mapping[oh][0]:mapping[ot][-1] + 1]
                ))
    return list(spoes)


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """

    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(tokenizer.tokenize(spo[2])),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


import ray

from fastapi import FastAPI
from ray import serve

app = FastAPI(
    title='xxxx ',
    description='xxx',
    version='1.0.0'
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Item(BaseModel):
    array: str


def get_model(model_path):
    # 加载预训练模型
    base = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        return_keras_model=False
    )

    # 预测结果
    entity_output = GlobalPointer(heads=2, head_size=64)(base.model.output)
    head_output = GlobalPointer(
        heads=len(predicate2id), head_size=64, RoPE=False, tril_mask=False
    )(base.model.output)
    tail_output = GlobalPointer(
        heads=len(predicate2id), head_size=64, RoPE=False, tril_mask=False
    )(base.model.output)
    outputs = [entity_output, head_output, tail_output]
    model = keras.models.Model(base.model.inputs, outputs)

    model.load_weights(model_path)
    return model


@serve.deployment(route_prefix="/")
@serve.ingress(app)
class MyFastAPIDeployment:
    def __init__(self, model_path):
        self.model = get_model(model_path)

    @app.post("/medical_gp_linker")
    async def root(self, item: Item):
        # Step 1: transform HTTP request -> tensorflow input
        # Here we define the request schema to be a json array.

        # Step 2: tensorflow input -> tensorflow output
        prediction = extract_spoes(item.array, model=self.model)
        out = []

        for i in prediction:
            out.append({"source": i[0], "target": i[2], 'rela': i[1], "type": 'resolved'})

        return {
            "prediction": out,
        }


if __name__ == '__main__':

    ray.init(num_cpus=4)
    serve.start(http_options=HTTPOptions(num_cpus=2, host="0.0.0.0", port="6006"))
    TRAINED_MODEL_PATH = '../model/best_model_gplinker_120.weights'
    MyFastAPIDeployment.deploy(TRAINED_MODEL_PATH)
    while True:
        time.sleep(5)
        print(serve.list_deployments())

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 12:29:19 2023

@author: lixinxian
"""

import torch.nn as nn
import os
import pickle 
import json
import pandas as pd 
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
import time
from itertools import chain
from sklearn import metrics
import torch.optim as optim
import datetime

def get_vocab(data_path):
    # 词表保存路径
    vocab_path = './vocab.pkl'
    # 第一次运行需要遍历训练集获取到标签字典，并存储成json文件保存，第二次运行即可直接载入json文件
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as fp:
            vocab = pickle.load(fp)
    else:
        json_data = []
        # 加载数据集
        with open(data_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                print(line)
                json_data.append(json.loads(line))
        # 建立词表字典，提前加入'PAD'和'UNK'
        # 'PAD'：在一个batch中不同长度的序列用该字符补齐
        # 'UNK'：当验证集或测试集出现词表以外的词时，用该字符代替
        vocab = {'PAD': 0, 'UNK': 1}
        # 遍历数据集，不重复取出所有字符，并记录索引
        for data in json_data:
            for word in data['text']:  # 获取实体标签，如'name'，'compan
                if word not in vocab:
                    vocab[word] = len(vocab)
        # vocab：{'PAD': 0, 'UNK': 1, '浙': 2, '商': 3, '银': 4, '行': 5...}
        # 保存成pkl文件
        with open(vocab_path, 'wb') as fp:
            pickle.dump(vocab, fp)

    # 翻转字表，预测时输出的序列为索引，方便转换成中文汉字
    # vocab_inv：{0: 'PAD', 1: 'UNK', 2: '浙', 3: '商', 4: '银', 5: '行'...}
    vocab_inv = {v: k for k, v in vocab.items()}
    return vocab, vocab_inv

def get_label_map(data_path):
    # 标签字典保存路径
    label_map_path = './label_map.json'
    # 第一次运行需要遍历训练集获取到标签字典，并存储成json文件保存，第二次运行即可直接载入json文件
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r', encoding='utf-8') as fp:
            label_map = json.load(fp)
    else:
        # 读取json数据
        json_data = []
        with open(data_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                json_data.append(json.loads(line))
        '''
        json_data[0]数据为该格式：

        {'text': '浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，',
         'label': {'name': {'叶老桂': [[9, 11]]}, 'company': {'浙商银行': [[0, 3]]}}}
        '''
        # 统计共有多少类别
        n_classes = []
        for data in json_data:
            for label in data['label'].keys():  # 获取实体标签，如'name'，'company'
                if label not in n_classes:  # 将新的标签加入到列表中
                    n_classes.append(label)
        n_classes.sort()

        # n_classes: ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
        # 设计label_map字典，对每个标签设计两种，如B-name、I-name，并设置其ID值
        label_map = {}
        for n_class in n_classes:
            label_map['B-' + n_class] = len(label_map)
            label_map['I-' + n_class] = len(label_map)
        label_map['O'] = len(label_map)
        # 对于BiLSTM+CRF网络，需要增加开始和结束标签，以增强其标签约束能力
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        label_map[START_TAG] = len(label_map)
        label_map[STOP_TAG] = len(label_map)
        '''
        {'B-address': 0, 'I-address': 1, 'B-book': 2, 'I-book': 3, 'B-company': 4, 'I-company': 5, 'B-game': 6,
         'I-game': 7, 'B-government': 8, 'I-government': 9, 'B-movie': 10, 'I-movie': 11, 'B-name': 12, 'I-name': 13,
         'B-organization': 14, 'I-organization': 15, 'B-position': 16, 'I-position': 17, 'B-scene': 18, 'I-scene': 19,
         'O': 20, '<START>': 21, '<STOP>': 22}
        '''
        
        # 将label_map字典存储成json文件
        with open(label_map_path, 'w', encoding='utf-8') as fp:
            json.dump(label_map, fp, indent=4)

    # {0: 'B-address', 1: 'I-address', 2: 'B-book', 3: 'I-book'...}
    label_map_inv = {v: k for k, v in label_map.items()}
    return label_map, label_map_inv

def get_label_map(data_path):
    # 标签字典保存路径
    label_map_path = './label_map.json'
    # 第一次运行需要遍历训练集获取到标签字典，并存储成json文件保存，第二次运行即可直接载入json文件
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r', encoding='utf-8') as fp:
            label_map = json.load(fp)
    else:
        # 读取json数据
        json_data = []
        with open(data_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                json_data.append(json.loads(line))
        '''
        json_data[0]数据为该格式：

        {'text': '浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，',
         'label': {'name': {'叶老桂': [[9, 11]]}, 'company': {'浙商银行': [[0, 3]]}}}
        '''
        # 统计共有多少类别
        n_classes = []
        for data in json_data:
            for label in data['label'].keys():  # 获取实体标签，如'name'，'company'
                if label not in n_classes:  # 将新的标签加入到列表中
                    n_classes.append(label)
        n_classes.sort()

        # n_classes: ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
        # 设计label_map字典，对每个标签设计两种，如B-name、I-name，并设置其ID值
        label_map = {}
        for n_class in n_classes:
            label_map['B-' + n_class] = len(label_map)
            label_map['I-' + n_class] = len(label_map)
        label_map['O'] = len(label_map)
        # 对于BiLSTM+CRF网络，需要增加开始和结束标签，以增强其标签约束能力
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        label_map[START_TAG] = len(label_map)
        label_map[STOP_TAG] = len(label_map)
        '''
        {'B-address': 0, 'I-address': 1, 'B-book': 2, 'I-book': 3, 'B-company': 4, 'I-company': 5, 'B-game': 6,
         'I-game': 7, 'B-government': 8, 'I-government': 9, 'B-movie': 10, 'I-movie': 11, 'B-name': 12, 'I-name': 13,
         'B-organization': 14, 'I-organization': 15, 'B-position': 16, 'I-position': 17, 'B-scene': 18, 'I-scene': 19,
         'O': 20, '<START>': 21, '<STOP>': 22}
        '''
        
        # 将label_map字典存储成json文件
        with open(label_map_path, 'w', encoding='utf-8') as fp:
            json.dump(label_map, fp, indent=4)

    # {0: 'B-address', 1: 'I-address', 2: 'B-book', 3: 'I-book'...}
    label_map_inv = {v: k for k, v in label_map.items()}
    return label_map, label_map_inv

def data_process(path):
    # 读取每一条json数据放入列表中
    # 由于该json文件含多个数据，不能直接json.loads读取，需使用for循环逐条读取
    json_data = []
    with open(path, 'r', encoding='utf-8') as fp:
        for line in fp:
            json_data.append(json.loads(line))

    data = []
    # 遍历json_data中每组数据
    for i in range(len(json_data)):
        # 将标签全初始化为'O'
        label = ['O'] * len(json_data[i]['text'])
        # 遍历'label'中几组实体，如样例中'name'和'company'
        for n in json_data[i]['label']:
            # 遍历实体中几组文本，如样例中'name'下的'叶老桂'（有多组文本的情况，样例中只有一组）
            for key in json_data[i]['label'][n]:
                # 遍历文本中几组下标，如样例中[[9, 11]]（有时某个文本在该段中出现两次，则会有两组下标）
                for n_list in range(len(json_data[i]['label'][n][key])):
                    # 记录实体开始下标和结尾下标
                    start = json_data[i]['label'][n][key][n_list][0]
                    end = json_data[i]['label'][n][key][n_list][1]
                    # 将开始下标标签设为'B-' + n，如'B-' + 'name'即'B-name'
                    # 其余下标标签设为'I-' + n
                    label[start] = 'B-' + n
                    label[start + 1: end + 1] = ['I-' + n] * (end - start)

        # 对字符串进行字符级分割
        # 英文文本如'bag'分割成'b'，'a'，'g'三位字符，数字文本如'125'分割成'1'，'2'，'5'三位字符
        texts = []
        for t in json_data[i]['text']:
            texts.append(t)

        # 将文本和标签编成一个列表添加到返回数据中
        data.append([texts, label])
    return data


class Mydataset(Dataset):
    def __init__(self, file_path, vocab, label_map):
        self.file_path = file_path
        # 数据预处理
        self.data = data_process(self.file_path)
        self.label_map, self.label_map_inv = label_map
        self.vocab, self.vocab_inv = vocab
        # self.data为中文汉字和英文标签，将其转化为索引形式
        self.examples = []
        for text, label in self.data:
            t = [self.vocab.get(t, self.vocab['UNK']) for t in text]
            l = [self.label_map[l] for l in label]
            self.examples.append([t, l])

    def __getitem__(self, item):
        return self.examples[item]

    def __len__(self):
        return len(self.data)

    def collect_fn(self, batch):
        # 取出一个batch中的文本和标签，将其单独放到变量中处理
        # 长度为batch_size，每个序列长度为原始长度
        text = [t for t, l in batch]
        label = [l for t, l in batch]
        # 获取一个batch内所有序列的长度，长度为batch_size
        seq_len = [len(i) for i in text]
        # 提取出最大长度用于填充
        max_len = max(seq_len)

        # 填充到最大长度，文本用'PAD'补齐，标签用'O'补齐
        text = [t + [self.vocab['PAD']] * (max_len - len(t)) for t in text]
        label = [l + [self.label_map['O']] * (max_len - len(l)) for l in label]

        # 将其转化成tensor，再输入到模型中，这里的dtype必须是long否则报错
        # text 和 label shape：(batch_size, max_len)
        # seq_len shape：(batch_size,)
        text = torch.tensor(text, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        seq_len = torch.tensor(seq_len, dtype=torch.long)
        
        return text, label, seq_len
    
def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


# log sum exp 增强数值稳定性
# 改进了torch版本原始函数.可适用于两种情况计算得分
def log_sum_exp(vec):
    max_score, _ = torch.max(vec, dim=-1)
    max_score_broadcast = max_score.unsqueeze(-1).repeat_interleave(vec.shape[-1], dim=-1)
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=-1))


class BiLSTM_CRF(nn.Module):
    def __init__(self, dataset, embedding_dim, hidden_dim, device='cpu'):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim  # 词向量维度
        self.hidden_dim = hidden_dim  # 隐层维度
        self.vocab_size = len(dataset.vocab)  # 词表大小
        self.tagset_size = len(dataset.label_map)  # 标签个数
        self.device = device
        # 记录状态，'train'、'eval'、'pred'对应三种不同的操作
        self.state = 'train'  # 'train'、'eval'、'pred'

        self.word_embeds = nn.Embedding(self.vocab_size, embedding_dim)
        # BiLSTM会将两个方向的输出拼接，维度会乘2，所以在初始化时维度要除2
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=2, bidirectional=True, batch_first=True)

        # BiLSTM 输出转化为各个标签的概率，此为CRF的发射概率
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size, bias=False)
        # 初始化CRF类
        self.crf = CRF(dataset, device)
        self.dropout = nn.Dropout(p=0.5, inplace=True)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

    def _get_lstm_features(self, sentence, seq_len):
        embeds = self.word_embeds(sentence)
        self.dropout(embeds)

        # 输入序列进行了填充，但RNN不能对填充后的'PAD'也进行计算，所以这里使用了torch自带的方法
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, seq_len, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        seq_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        seqence_output = self.layer_norm(seq_unpacked)
        lstm_feats = self.hidden2tag(seqence_output)
        return lstm_feats

    def forward(self, sentence, tags, seq_len):
        # 输入序列经过BiLSTM得到发射概率
        feats = self._get_lstm_features(sentence, seq_len)
        # 根据 state 判断哪种状态，从而选择计算损失还是维特比得到预测序列
        if self.state == 'train':
            loss = self.crf.neg_log_likelihood(feats, tags, seq_len)
            return loss
        else:
            all_tag = []
            for i, feat in enumerate(feats):
                # path_score, best_path = self.crf._viterbi_decode(feat[:seq_len[i]])
                all_tag.append(self.crf._viterbi_decode(feat[:seq_len[i]])[1])
            return all_tag


class CRF:
    def __init__(self, dataset, device='cpu'):
        self.label_map = dataset.label_map
        self.label_map_inv = dataset.label_map_inv
        self.tagset_size = len(self.label_map)
        self.device = device

        # 转移概率矩阵
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)).to(self.device)

        # 增加开始和结束标志，并手动干预转移概率
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        self.transitions.data[self.label_map[self.START_TAG], :] = -10000
        self.transitions.data[:, self.label_map[self.STOP_TAG]] = -10000

    def _forward_alg(self, feats, seq_len):
        # 手动设置初始得分，让开始标志到其他标签的得分最高
        init_alphas = torch.full((self.tagset_size,), -10000.)
        init_alphas[self.label_map[self.START_TAG]] = 0.

        # 记录所有时间步的得分，为了解决序列长度不同问题，后面直接取各自长度索引的得分即可
        # shape：(batch_size, seq_len + 1, tagset_size)
        forward_var = torch.zeros(feats.shape[0], feats.shape[1] + 1, feats.shape[2], dtype=torch.float32,
                                  device=self.device)
        forward_var[:, 0, :] = init_alphas

        # 将转移概率矩阵复制 batch_size 次，批次内一起进行计算，矩阵计算优化，加快运行效率
        # shape：(batch_size, tagset_size) -> (batch_size, tagset_size, tagset_size)
        transitions = self.transitions.unsqueeze(0).repeat(feats.shape[0], 1, 1)
        # 对所有时间步进行遍历
        for seq_i in range(feats.shape[1]):
            # 取出当前词发射概率
            emit_score = feats[:, seq_i, :]
            # 前一时间步得分 + 转移概率 + 当前时间步发射概率
            tag_var = (
                    forward_var[:, seq_i, :].unsqueeze(1).repeat(1, feats.shape[2], 1)  # (batch_size, tagset_size, tagset_size)
                    + transitions
                    + emit_score.unsqueeze(2).repeat(1, 1, feats.shape[2])
            )
            # 这里必须调用clone，不能直接在forward_var上修改，否则在梯度回传时会报错
            cloned = forward_var.clone()
            cloned[:, seq_i + 1, :] = log_sum_exp(tag_var)
            forward_var = cloned

        # 按照不同序列长度不同取出最终得分
        forward_var = forward_var[range(feats.shape[0]), seq_len, :]
        # 手动干预,加上结束标志位的转移概率
        terminal_var = forward_var + self.transitions[self.label_map[self.STOP_TAG]].unsqueeze(0).repeat(feats.shape[0], 1)
        # 得到最终所有路径的分数和
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags, seq_len):
        # 初始化,大小为(batch_size,)
        score = torch.zeros(feats.shape[0], device=self.device)
        # 将开始标签拼接到序列上起始位置，参与分数计算
        start = torch.tensor([self.label_map[self.START_TAG]], device=self.device).unsqueeze(0).repeat(feats.shape[0], 1)
        tags = torch.cat([start, tags], dim=1)
        # 在batch上遍历
        for batch_i in range(feats.shape[0]):
            # 采用矩阵计算方法，加快运行效率
            # 取出当前序列所有时间步的转移概率和发射概率进行相加，由于计算真实标签序列的得分，所以只选择标签的路径
            score[batch_i] = torch.sum(
                self.transitions[tags[batch_i, 1:seq_len[batch_i] + 1], tags[batch_i, :seq_len[batch_i]]]) \
                             + torch.sum(feats[batch_i, range(seq_len[batch_i]), tags[batch_i][1:seq_len[batch_i] + 1]])
            # 最后加上结束标志位的转移概率
            score[batch_i] += self.transitions[self.label_map[self.STOP_TAG], tags[batch_i][seq_len[batch_i]]]
        return score

    # 维特比算法得到最优路径,原始torch函数
    def _viterbi_decode(self, feats):
        backpointers = []

        # 手动设置初始得分，让开始标志到其他标签的得分最高
        init_vvars = torch.full((1, self.tagset_size), -10000., device=self.device)
        init_vvars[0][self.label_map[self.START_TAG]] = 0

        # 用于记录前一时间步的分数
        forward_var = init_vvars
        # 传入的就是单个序列,在每个时间步上遍历
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            # 一个标签一个标签去计算处理
            for next_tag in range(self.tagset_size):
                # 前一时间步分数 + 转移到第 next_tag 个标签的概率
                next_tag_var = forward_var + self.transitions[next_tag]
                # 得到最大分数所对应的索引,即前一时间步哪个标签过来的分数最高
                best_tag_id = argmax(next_tag_var)
                # 将该索引添加到路径中
                bptrs_t.append(best_tag_id)
                # 将此分数保存下来
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # 在这里加上当前时间步的发射概率，因为之前计算每个标签的最大分数来源与当前时间步发射概率无关
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            # 将当前时间步所有标签最大分数的来源索引保存
            backpointers.append(bptrs_t)

        # 手动加入转移到结束标签的概率
        terminal_var = forward_var + self.transitions[self.label_map[self.STOP_TAG]]
        # 在最终位置得到最高分数所对应的索引
        best_tag_id = argmax(terminal_var)
        # 最高分数
        path_score = terminal_var[0][best_tag_id]

        # 回溯，向后遍历得到最优路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # 弹出开始标签
        start = best_path.pop()
        assert start == self.label_map[self.START_TAG]  # Sanity check
        # 将路径反转
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, feats, tags, seq_len):
        # 所有路径得分
        forward_score = self._forward_alg(feats, seq_len)
        # 标签路径得分
        gold_score = self._score_sentence(feats, tags, seq_len)
        # 返回 batch 分数的平均值
        return torch.mean(forward_score - gold_score)
    
    
def train(model, optimizer):
    total_start = time.time()
    best_score = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        model.state = 'train'
        for step, (text, label, seq_len) in enumerate(train_dataloader, start=1):
            start = time.time()
            text = text.to(device)
            label = label.to(device)
            #seq_len = seq_len.to(device)
            seq_len = seq_len.to("cpu")
            
            loss = model(text, label, seq_len)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f'Epoch: [{epoch + 1}/{epochs}],'
                  f'  cur_epoch_finished: {step * batch_size / len(train_dataset) * 100:2.2f}%,'
                  f'  loss: {loss.item():2.4f},'
                  f'  cur_step_time: {time.time() - start:2.2f}s,'
                  f'  cur_epoch_remaining_time: {datetime.timedelta(seconds=int((len(train_dataloader) - step) / step * (time.time() - epoch_start)))}',
                  f'  total_remaining_time: {datetime.timedelta(seconds=int((len(train_dataloader) * epochs - (len(train_dataloader) * epoch + step)) / (len(train_dataloader) * epoch + step) * (time.time() - total_start)))}')

        # 每周期验证一次，保存最优参数
        score = evaluate()
        if score > best_score:
            print(f'score increase:{best_score} -> {score}')
            best_score = score
            torch.save(model.state_dict(), './model.bin')
        print(f'current best score: {best_score}')


def evaluate():
    # model.load_state_dict(torch.load('./model1.bin'))
    all_label = []
    all_pred = []
    model.eval()
    model.state = 'eval'
    with torch.no_grad():
        for text, label, seq_len in tqdm(valid_dataloader, desc='eval: '):
            text = text.to(device)
            #seq_len = seq_len.to(device)
            seq_len = seq_len.to("cpu")
            batch_tag = model(text, label, seq_len)
            all_label.extend([[train_dataset.label_map_inv[t] for t in l[:seq_len[i]].tolist()] for i, l in enumerate(label)])
            all_pred.extend([[train_dataset.label_map_inv[t] for t in l] for l in batch_tag])

    all_label = list(chain.from_iterable(all_label))
    all_pred = list(chain.from_iterable(all_pred))
    sort_labels = [k for k in train_dataset.label_map.keys()]
    # 使用sklearn库得到F1分数
    f1 = metrics.f1_score(all_label, all_pred, average='macro', labels=sort_labels[:-3])

    print(metrics.classification_report(
        all_label, all_pred, labels=sort_labels[:-3], digits=3
    ))
    return f1

def read_json():
    path = "./datasets/CREDIT_DISPOSITION.json";
    f = open(path, 'r', encoding = 'utf-8');
    content = f.read()
    json_ = json.loads(content);
    f.close();
    return json_;

def find_pos(str_, s):
    pos_ = [];
    start_len = 0;
    while True:
        num = str_.find(s,start_len)
        if num == -1:  # 找不到则返回-1
            break
        pos_.append([num, num + len(s)]);
        start_len = num + 1;
    return pos_;

#replace一些特殊字符
def rep_spe_char(text):
    text = text.replace("\xa0", "");
    text = text.replace("\t", "");
    text = text.replace(" ", "");
    return text;

def dealt_json_to_split(jsp):
    text = jsp["data"]["text"];
    annlist = jsp["annotations"];
    dictlist = [];
    #获取关键词所在句子,并摘取出来
    for annele in annlist:
        res = annele["result"];
        for r in res:
            v = r["value"];
            dictlist.append(v);
    text = rep_spe_char(text);
    textlists = text.split("////");
    #对dictlist中的内容进行提纯
    dict_ = [dic["text"] for dic in dictlist];
    diclabels_ = [dic["labels"][0] for dic in dictlist];
# =============================================================================
#     #对于textlists中的每个句子,主要包含dict_中的元素,则进行打标签BI,如果不包含按照长度打成O
#     textlists = ["彭小军认为，国内银行现在走的是台湾的发卡模式，先通过跑马圈地再在圈的地里面选择客户，"];
#     dict_ = ["台湾", "彭小军"];
#     diclabels_ = ["address", "name"];
# =============================================================================
    tetlablelists = [];
    for tet in textlists:
        tetlabellist = {};
        tetlabellist["text"] = tet;
        tetlabellist["label"] = {};
        for idx in range(len(dict_)):
            str_ = rep_spe_char(dict_[idx]);
            ss = find_pos(tet, str_);
            #对于tet来说,如果ss标记了,则把相应的位置替换掉之后再进行第二个关键词判断
            if(len(ss) > 0):
                tetlabellist["label"][diclabels_[idx]] = {};
                for s in ss:
                    bstart = s[0];
                    iend = s[1] - 1;
                    tetlabellist["label"][diclabels_[idx]][dict_[idx]] = [[bstart, iend]]
        #将tet和tetlabellist装在df里
        if tetlabellist["label"]:
            tetlablelists.append(tetlabellist);
    return tetlablelists;

def split_train_test(data, test_ratio):
    #设置随机数种子，保证每次生成的结果都是一样的
    np.random.seed(42)
    #permutation随机生成0-len(data)随机序列
    shuffled_indices = np.random.permutation(len(data))
    #test_ratio为测试集所占的半分比
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return [data[i] for i in train_indices], [data[i] for i in test_indices];

def squeeze_train_dev(json_):
    rrlist = [];
    for jsp in json_:
        rr = dealt_json_to_split(jsp);
        rrlist.extend(rr);
    print(rrlist);
    rrlist = np.array(rrlist);
    train_set,test_set = split_train_test(rrlist, 0.2)
    #将train_set写入train.json; 将test_set写入test.json
    train_file = open('./train.json', 'w+', encoding = "utf-8");
    for i in train_set:
        train_file.write(json.dumps(i));
        train_file.write("\n");
    train_file.close();
    test_file = open('./dev.json', 'w+', encoding = "utf-8");
    for i in test_set:
        test_file.write(json.dumps(i));
        test_file.write("\n");
    test_file.close();

if __name__ == "__main__":
    #将原始数据切割成 {text: 'xxxxxxxx', label: {COM: {xxx公司: [[startpos, endpos]]}}}
    #读取jsona
    json_ = read_json();
    squeeze_train_dev(json_);
    embedding_size = 128
    hidden_dim = 768
    epochs = 100
    batch_size = 32
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 建立中文词表，扫描训练集所有字符得到，'PAD'在batch填充时使用，'UNK'用于替换字表以外的新字符
    vocab = get_vocab('./train.json')
    # 建立标签字典，扫描训练集所有字符得到
    label_map = get_label_map('./train.json')

    train_dataset = Mydataset('./train.json', vocab, label_map)
    valid_dataset = Mydataset('./dev.json', vocab, label_map)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True,
                                  collate_fn=train_dataset.collect_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, pin_memory=False, shuffle=False,
                                  collate_fn=valid_dataset.collect_fn)
    

    model = BiLSTM_CRF(train_dataset, embedding_size, hidden_dim, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    train(model, optimizer)
    
#打标 + 学习一下模型 + 自己搜搜相关论文,熟悉不同刊的论文结构(计算机顶刊,金融顶刊RFS) + 论文码字 + 找一下文本源
#下一阶段的难点: 解决两个模型的预测精确度: 解决一下Bert的embedding + 打标 + 找一下文本源

#找文本源+达标(ddl: );
#搜论文(ddl: 一周内)+码字;
#学习模型(ddl: );
#提升模型精度(ddl: 一周时间); 

#论文工作量体现: 
#  模型: 基于Bert  基于bil 基于bili-crf ===>
#  金融图谱
#===============================

"""
AMC公司      B公司       什么事    时间     金额
-----> neo4j
"""


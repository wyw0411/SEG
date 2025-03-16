import os

import torch
import torch.nn as nn
import sys
new_path = '/home/user/.local/lib/python3.6/site-packages'
from config import DEVICE
from transformers import BertTokenizer  # 分词器

sys.path.append(new_path)
from thulac import thulac
import numpy as np

from data.emotion_dictory import config

class EmotionPolarity(nn.Module):
    def __init__(self, configs):
        super(EmotionPolarity, self).__init__()

        self.emo_emb = nn.Embedding(7, 768)
        self.flag_emb = nn.Embedding(2, 768)

        self.spo_file_paths = '../data/emotion_dictory/emo_class.txt'
        self.lookup_table = self._create_lookup_table()
        self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG
        self.tokenizer = thulac(seg_only=True, user_dict='../data/emotion_dictory/emo_dictory.txt')
        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)

    def forward(self, doc_str, bert_token_b, max_doc_len):
        bs_len = len(doc_str)
        assert bs_len == bert_token_b.size()[0]

        emo_class_list = []
        emo_str_list = []
        # print("doc_str:{}".format(doc_str))
        for bs in range(bs_len):
            doc_str_b = doc_str[bs]
            split_sent_clause = [self.tokenizer.cut(doc_str_b.strip())]
            # print("split_sent_clause:{}".format(split_sent_clause))
            emo_class_list_b = []
            emo_str_list_b = []
            for split_sent in split_sent_clause:
                sent_id = 0
                # emo_class = []
                emo_str = []
                for token in split_sent:
                    if(token[0] == ' '):
                        continue
                    if token[0] == '[CLS]':
                        sent_id += 1
                        emo_class_list_b.append(0)
                        continue
                    if token[0] == '[SEP]':
                        emo_class_list_b.append(0)
                        continue
                    idx = self.lookup_table.get(token[0], [])
                    if idx == []:
                        length = len(token[0])
                        for i in range(length):
                            if 48 <= ord(token[0][i]) <= 57:
                                length -= 1
                        for i in range(length):
                            emo_class_list_b.append(0)
                    else:
                        for i in range(len(token[0])):
                            emo_class_list_b.append(int(idx))
                        emo_str.append([sent_id, token[0], int(idx)])  # 这里的sent_id是从1开始，不是从0开始
                emo_str_list_b.append(emo_str)
            emo_class_list.append(emo_class_list_b)
            emo_str_list.append(emo_str_list_b)


        detect_emo = np.zeros((bs_len, max_doc_len), dtype=int)
        for b in range(bs_len):
            emo_str = emo_str_list[b][0]
            for sent in emo_str:
                sent_id = sent[0]  # 从1开始
                # print(sent_id)
                sent_type = sent[2]
                if sent_id - 1 < max_doc_len:
                    if detect_emo[b][sent_id - 1] == 0:
                        detect_emo[b][sent_id - 1] = sent_type
                    else:
                        continue
                else:
                    continue
        emo_clause_emb = torch.zeros(bs_len, max_doc_len, 768)
        for b in range(bs_len):
            detect_emo_b = detect_emo[b]
            for idx, j in enumerate(detect_emo_b):
                emo_clause_emb[b][idx] = self.emo_emb(torch.tensor(j).to(DEVICE))
        return emo_clause_emb.to(DEVICE), emo_class_list, emo_str_list



    def _create_lookup_table(self):
        lookup_table = {}
        print("[Emotion_Dic] Loading txt from {}".format(self.spo_file_paths))
        f = open(self.spo_file_paths, 'r', encoding='utf-8')
        conts = f.readlines()
        for line in conts:
            a_line = line.strip('\n').split(' ')
            subj, idx = a_line[0], a_line[1]
            lookup_table[subj] = idx
        return lookup_table



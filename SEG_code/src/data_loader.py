import sys
sys.path.append('..')
from os.path import join
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence  # 填充序列
from transformers import BertTokenizer  # 分词器
from config import *
from src.utils.utils import *
import tqdm


torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
torch.backends.cudnn.deterministic = True


def build_train_data(configs, fold_id, shuffle=True):
    train_dataset = MyDataset(configs, fold_id, data_type='train')

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=shuffle, collate_fn=bert_batch_preprocessing)
    return train_loader


def build_inference_data(configs, fold_id, data_type):
    dataset = MyDataset(configs, fold_id, data_type)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=configs.batch_size,
                                              shuffle=False, collate_fn=bert_batch_preprocessing)

    # return np.array(doc_len_b), np.array(adj_b), \doc_id_b
    #        np.array(y_emotions_b), np.array(y_causes_b), np.array(y_mask_b), doc_couples_b, doc_id_b, \
    #        bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b
    return data_loader


class MyDataset(Dataset):
    def __init__(self, configs, fold_id, data_type, data_dir=DATA_DIR):
        self.data_dir = data_dir  # '../data'
        self.split = configs.split  # split10,十次交叉验证
        # self.multi = 'multi'
        self.data_type = data_type
        self.train_file = join(data_dir, self.split, TRAIN_FILE % fold_id)

        self.valid_file = join(data_dir, self.split, VALID_FILE % fold_id)
        self.test_file = join(data_dir, self.split, TEST_FILE % fold_id)
        # self.test_file = join(data_dir, self.multi, MULTI_TEST_FILE % fold_id)

        # print(self.train_file)

        self.batch_size = configs.batch_size  # 2
        self.epochs = configs.epochs  # 20

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)

        self.doc_couples_list, self.y_emotions_list, self.y_causes_list, \
        self.doc_len_list, self.doc_id_list, \
        self.bert_token_idx_list, self.bert_clause_idx_list, self.bert_segments_idx_list, \
        self.bert_token_lens_list, self.bert_sep_idx_list, self.doc_str, self.emotion_type = self.read_data_file(self.data_type)
        # print('doc_couples_list:{},y_emotions_list:{},y_causes_list:{},doc_len_list:{},doc_id_list:{},\
        #       bert_token_idx_list{},bert_clause_idx_list:{},bert_segment_idx:{},bert_token_lens_list:{}'\
        #     .format (self.doc_couples_list,self.y_emotions_list,self.y_causes_list,self.doc_len_list,\
        #      self.doc_id_list,self.bert_token_idx_list,self.bert_clause_idx_list, \
        #      self.bert_segments_idx_list,self.bert_token_lens_list))


    def __len__(self):
        return len(self.y_emotions_list)

    def __getitem__(self, idx):
        doc_couples, y_emotions, y_causes = self.doc_couples_list[idx], self.y_emotions_list[idx], self.y_causes_list[idx]
        doc_len, doc_id = self.doc_len_list[idx], self.doc_id_list[idx]
        bert_token_idx, bert_clause_idx, bert_sep_idx = self.bert_token_idx_list[idx], self.bert_clause_idx_list[idx], self.bert_sep_idx_list[idx]
        bert_segments_idx, bert_token_lens = self.bert_segments_idx_list[idx], self.bert_token_lens_list[idx]
        doc_str = self.doc_str[idx]
        emotion_type = self.emotion_type[idx]

        if bert_token_lens > 512:
            bert_token_idx, bert_clause_idx, \
            bert_segments_idx, bert_token_lens, \
            doc_couples, y_emotions, y_causes, doc_len, bert_sep_idx, doc_str = self.token_trunk(bert_token_idx, bert_clause_idx,
                                                                          bert_segments_idx, bert_token_lens,
                                                                          doc_couples, y_emotions, y_causes, doc_len, bert_sep_idx, doc_str)

            # if int(doc_id) == 1577:
            #     print('found')
            #     print("1577:{}".format(doc_str))
            #     print(len(doc_str))
        bert_token_idx = torch.LongTensor(bert_token_idx)
        bert_segments_idx = torch.LongTensor(bert_segments_idx)
        bert_clause_idx = torch.LongTensor(bert_clause_idx)
        bert_sep_idx = torch.LongTensor(bert_sep_idx)

        assert doc_len == len(y_emotions)
        return doc_couples, y_emotions, y_causes, doc_len, doc_id, \
               bert_token_idx, bert_clause_idx, bert_segments_idx, bert_token_lens, bert_sep_idx, doc_str, emotion_type

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'valid':
            data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file

        doc_id_list = []
        doc_len_list = []
        doc_couples_list = []
        y_emotions_list, y_causes_list = [], []
        bert_token_idx_list = []
        bert_clause_idx_list = []
        bert_sep_idx_list = []
        bert_segments_idx_list = []
        bert_token_lens_list = []
        doc_str_list = []
        emotion_type = []

        data_list = read_json(data_file)

        for doc in data_list:
            doc_id = doc['doc_id']
            doc_len = doc['doc_len']
            doc_couples = doc['pairs']
            doc_emotions, doc_causes = zip(*doc_couples)

            doc_id_list.append(doc_id)
            doc_len_list.append(doc_len)
            doc_couples = list(map(lambda x: list(x), doc_couples))

            doc_couples_list.append(doc_couples)

            y_emotions, y_causes = [], []
            doc_clauses = doc['clauses']
            doc_str = ''
            sub_emotion_type = []
            for i in range(doc_len):
                emotion_label = int(i + 1 in doc_emotions)  # doc_emotions存放的是每个文档的情绪序号
                # 为什么要+1，因为range是从0开始
                cause_label = int(i + 1 in doc_causes)
                y_emotions.append(emotion_label)
                y_causes.append(cause_label)
                # 取出每个文档的c和e的序号
                clause = doc_clauses[i]

                clause_id = clause['clause_id']  # 第i个子句的id(i+1)
                if clause['emotion_category'] != "null":
                    emotion = clause['emotion_category']
                    if emotion == "happiness":
                        #sub_emotion_type = [1, 0, 0, 0, 0, 0]
                        sub_emotion_type.append([1, i])
                    elif emotion == "sadness":
                        sub_emotion_type.append([2, i])
                    elif emotion == "fear":
                        sub_emotion_type.append([3, i])
                    elif emotion == "anger":
                        sub_emotion_type.append([4, i])
                    elif emotion == "disgust":
                        sub_emotion_type.append([5, i])
                    elif emotion == "surprise":
                        sub_emotion_type.append([6, i])


                assert int(clause_id) == i + 1
                # 若为false则触发异常
                doc_str += '[CLS] ' + clause['clause'] + ' [SEP] '


            indexed_tokens = self.bert_tokenizer.encode(doc_str.strip(), add_special_tokens=False)
            # print('indeded_tokens:{}'.format(indexed_tokens))

            clause_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
            sep_indices = [i for i, x in enumerate(indexed_tokens) if x == 102]
            # print('clause_indices:{}'.format(clause_indices))
            doc_token_len = len(indexed_tokens)

            segments_ids = []
            segments_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
            segments_indices.append(len(indexed_tokens))

            for i in range(len(segments_indices)-1):
                semgent_len = segments_indices[i+1] - segments_indices[i]
                if i % 2 == 0:
                    segments_ids.extend([0] * semgent_len)
                else:
                    segments_ids.extend([1] * semgent_len)

            assert len(clause_indices) == doc_len
            assert len(sep_indices) == doc_len
            assert len(segments_ids) == len(indexed_tokens)
            bert_token_idx_list.append(indexed_tokens)
            bert_clause_idx_list.append(clause_indices)
            bert_segments_idx_list.append(segments_ids)
            bert_token_lens_list.append(doc_token_len)
            bert_sep_idx_list.append(sep_indices)
            emotion_type.append(sub_emotion_type)

            y_emotions_list.append(y_emotions)
            y_causes_list.append(y_causes)
            doc_str_list.append(doc_str)





        return doc_couples_list, y_emotions_list, y_causes_list, doc_len_list, doc_id_list, \
               bert_token_idx_list, bert_clause_idx_list, bert_segments_idx_list, bert_token_lens_list, bert_sep_idx_list, doc_str_list, emotion_type

    def token_trunk(self, bert_token_idx, bert_clause_idx, bert_segments_idx, bert_token_lens,
                    doc_couples, y_emotions, y_causes, doc_len, bert_sep_idx, doc_str):
        # TODO: cannot handle some extreme cases now  # 解决长度大于512的情况
        emotion, cause = doc_couples[0]
        if emotion > doc_len / 2 and cause > doc_len / 2:
            i = 0
            while True:
                temp_bert_token_idx = bert_token_idx[bert_clause_idx[i]:]
                if len(temp_bert_token_idx) <= 512:
                    cls_idx = bert_clause_idx[i]
                    sep_idx = bert_sep_idx[i]
                    bert_token_idx = bert_token_idx[cls_idx:]
                    doc_str = doc_str[cls_idx:]
                    bert_segments_idx = bert_segments_idx[cls_idx:]
                    bert_clause_idx = [p - cls_idx for p in bert_clause_idx[i:]]
                    bert_sep_idx = [p - cls_idx for p in bert_sep_idx[i:]]
                    doc_couples = [[emotion - i, cause - i]]
                    y_emotions = y_emotions[i:]
                    y_causes = y_causes[i:]
                    doc_len = doc_len - i
                    break
                i = i + 1
        if emotion < doc_len / 2 and cause < doc_len / 2:
            i = doc_len - 1
            while True:
                temp_bert_token_idx = bert_token_idx[:bert_clause_idx[i]]
                if len(temp_bert_token_idx) <= 512:
                    cls_idx = bert_clause_idx[i]
                    bert_token_idx = bert_token_idx[:cls_idx]
                    doc_str = doc_str[:cls_idx]
                    bert_segments_idx = bert_segments_idx[:cls_idx]
                    bert_clause_idx = bert_clause_idx[:i]
                    bert_sep_idx = bert_sep_idx[:i]
                    y_emotions = y_emotions[:i]
                    y_causes = y_causes[:i]
                    doc_len = i
                    break
                i = i - 1
        return bert_token_idx, bert_clause_idx, bert_segments_idx, bert_token_lens, \
               doc_couples, y_emotions, y_causes, doc_len, bert_sep_idx, doc_str


def bert_batch_preprocessing(batch):
    doc_couples_b, y_emotions_b, y_causes_b, doc_len_b, doc_id_b, \
    bert_token_b, bert_clause_b, bert_segment_b, bert_token_lens_b, bert_sep_b, doc_str, emotion_type = zip(*batch)


    y_mask_b, y_emotions_b, y_causes_b = pad_docs(doc_len_b, y_emotions_b, y_causes_b)
    adj_b = pad_matrices(doc_len_b)

    bert_token_b = pad_sequence(bert_token_b, batch_first=True, padding_value=0)  # true为在每一行填充0，false为将元素竖放，
    bert_segment_b = pad_sequence(bert_segment_b, batch_first=True, padding_value=0)
    bert_clause_b = pad_sequence(bert_clause_b, batch_first=True, padding_value=0)
    bert_sep_b = pad_sequence(bert_sep_b, batch_first=True, padding_value=0)
    # print(type(bert_token_b))
    bsz, max_len = bert_token_b.size()
    bert_global_masks_b = np.zeros([bsz, max_len], dtype=np.float)
    bert_local_masks_b = np.zeros([bsz, max_len], dtype=np.float)

    for bs_id in range(bsz):
        for clause_id in range(doc_len_b[bs_id]):
            cls_id = bert_clause_b[bs_id][clause_id]
            sep_id = bert_sep_b[bs_id][clause_id]
            bert_local_masks_b[bs_id][cls_id : sep_id+1] = 1


    bs = len(doc_len_b)
    max_doc_len = max(doc_len_b)
    max_seq_len = 0
    for i in range(bs):
        for j in range(doc_len_b[i]):
            max_seq_len = max(max_seq_len, bert_sep_b[i][j]-bert_clause_b[i][j]+1)

    bert_local_token_b = []
    bert_local_segment_b = []

    bert_global_masks_b = torch.FloatTensor(bert_global_masks_b)  # 变成tensor格式  [bs, seq_len]
    bert_local_masks_b = torch.FloatTensor(bert_local_masks_b)  # 变成tensor格式  [bs, seq_len]

    # assert bert_local_token_b.shape == bert_local_masks_b.shape
    # assert bert_local_segment_b.shape == bert_local_masks_b.shape
    assert bert_segment_b.shape == bert_token_b.shape
    assert bert_segment_b.shape == bert_global_masks_b.shape
    assert bert_global_masks_b.shape == bert_local_masks_b.shape


    return np.array(doc_len_b), np.array(adj_b), \
           np.array(y_emotions_b), np.array(y_causes_b), np.array(y_mask_b), doc_couples_b, doc_id_b, \
           bert_token_b, bert_segment_b, bert_global_masks_b, bert_local_masks_b, bert_local_token_b, bert_local_segment_b, bert_clause_b, bert_sep_b, doc_str, np.array(emotion_type)


def pad_docs(doc_len_b, y_emotions_b, y_causes_b):
    max_doc_len = max(doc_len_b)

    y_mask_b, y_emotions_b_, y_causes_b_ = [], [], []
    for y_emotions, y_causes in zip(y_emotions_b, y_causes_b):
        y_emotions_ = pad_list(y_emotions, max_doc_len, -1)  # 用-1填充使得y_emotions_长度为max_doc_len
        y_causes_ = pad_list(y_causes, max_doc_len, -1)
        y_mask = list(map(lambda x: 0 if x == -1 else 1, y_emotions_))  # 0进行mask，1不进行，这里可能是为了防止

        y_mask_b.append(y_mask)
        y_emotions_b_.append(y_emotions_)
        y_causes_b_.append(y_causes_)

    return y_mask_b, y_emotions_b_, y_causes_b_  # 长度均为max_doc_len


def pad_matrices(doc_len_b):
    N = max(doc_len_b)
    adj_b = []
    for doc_len in doc_len_b:
        adj = np.ones((doc_len, doc_len))
        adj = sp.coo_matrix(adj)  # 转为coo_matrix从而可以将data,row,col取出成为字符串形式
        adj = sp.coo_matrix((adj.data, (adj.row, adj.col)),
                            shape=(N, N), dtype=np.float32)
        adj_b.append(adj.toarray())
    return adj_b


def pad_list(element_list, max_len, pad_mark):
    element_list_pad = element_list[:]
    pad_mark_list = [pad_mark] * (max_len - len(element_list))
    element_list_pad.extend(pad_mark_list)
    return element_list_pad


import os.path
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE
from transformers import BertModel
import dgl
from dgl import load_graphs
from RelGraph import RelGraphConvLayer
from emo_detect import EmotionPolarity


class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.bert = BertModel.from_pretrained(configs.bert_cache_path)
        self.pred = Pre_Predictions(configs)
        self.pairwise_loss = configs.pairwise_loss
        self.rank = RankNN(configs)

        self.gcn_layers = 3
        self.rel_name_lists = [str(i) for i in range(5)]
        activation_func = nn.ReLU()
        self.hidden_dropout_prob = 0.1
        self.gcn = RelGraphConvLayer(configs.feat_dim, configs.feat_dim, self.rel_name_lists,
                                     num_bases=len(self.rel_name_lists),
                                     activation=activation_func, self_loop=True,
                                     dropout=self.hidden_dropout_prob * 3)
        self.GCN_layers = nn.ModuleList([self.gcn
                                         for i in range(self.gcn_layers)])
        self.middle_layer = nn.Sequential(
            nn.Linear(configs.feat_dim * (self.gcn_layers + 1), configs.feat_dim),
            activation_func,
            nn.Dropout(self.hidden_dropout_prob)
        )

        # ===emo_class===
        self.hidden = configs.hidden
        self.n_layers = configs.N
        self.d_model = configs.d_model
        self.dropout = configs.dp
        self.attn_heads = configs.att_heads
        self.emo_class = EmotionPolarity(configs)



    def forward(self, bert_token_b, bert_segment_b, bert_global_masks_b, bert_local_masks_b, bert_local_token_b,
                bert_local_segment_b, bert_clause_b, bert_sep_b, doc_len, adj, fold_id, epoch, doc_id, doc_str, mask, emotion_type, y_emotions_b, doc_couples_b):

        bs, seq_len = bert_token_b.size()

        #  ************global Bert**************
        bert_output = self.bert(input_ids=bert_token_b.to(DEVICE),
                                attention_mask=bert_global_masks_b.to(DEVICE),
                                token_type_ids=bert_segment_b.to(DEVICE))
        bert_output = bert_output[0]
        max_doc_len = bert_clause_b.size()[1]

        # ===emo_class===
        emo_clause_emb, emo_class_list, emo_str_list = self.emo_class(doc_str, bert_token_b, max_doc_len)


        #  ******local graph********
        final_local_graph_feature = self.local_graph(bert_output, doc_id, seq_len)

        local_feature = final_local_graph_feature + bert_output

        # *******fusion*********
        doc_local_sents_h = self.batched_index_select(local_feature, bert_clause_b.to(DEVICE), bert_sep_b.to(DEVICE), doc_len)
        bert_doc_output = self.batched_index_select(bert_output, bert_clause_b.to(DEVICE), bert_sep_b.to(DEVICE), doc_len)

        doc_sents_h = bert_doc_output + doc_local_sents_h + emo_clause_emb

        pred_e, pred_c = self.pred(doc_sents_h)
        couples_pred, emo_cau_pos, couple = self.rank(doc_sents_h)

        return couples_pred, emo_cau_pos, pred_e, pred_c


    def batched_index_select(self, bert_output, bert_clause_b, bert_sep_b, doc_len):
        dummy = bert_clause_b.unsqueeze(2).expand(bert_clause_b.size(0), bert_clause_b.size(1), bert_output.size(2))
        doc_sents_h = bert_output.gather(1, dummy)
        bs = bert_output.size()[0]
        # doc_sents_h = torch.randn(bs, bert_clause_b.size()[1], 768)
        for i in range(bs):
            for j in range(doc_len[i]):
                cls = bert_clause_b[i][j]
                sep = bert_sep_b[i][j]
                word_embedding_by_sentence = bert_output[i][cls + 1: sep]
                doc_sents_h[i][j] = torch.mean(word_embedding_by_sentence, dim=0)

        return doc_sents_h.to(DEVICE)

    def couple_generator(self, H, k):
        batch, seq_len, feat_dim = H.size()
        P_left = torch.cat([H] * seq_len, dim=2)
        P_left = P_left.reshape(-1, seq_len * seq_len, feat_dim)  # 包含两个位置的信息
        P_right = torch.cat([H] * seq_len, dim=1)  # 包含一个位置的信息
        P = torch.cat([P_left, P_right], dim=2)

        base_idx = np.arange(1, seq_len + 1)
        emo_pos = np.concatenate([base_idx.reshape(-1, 1)] * seq_len, axis=1).reshape(1, -1)[0]
        cau_pos = np.concatenate([base_idx] * seq_len, axis=0)

        rel_pos = cau_pos - emo_pos  # 相对位置
        rel_pos = torch.LongTensor(rel_pos).to(DEVICE)
        emo_pos = torch.LongTensor(emo_pos).to(DEVICE)
        cau_pos = torch.LongTensor(cau_pos).to(DEVICE)

        if seq_len > k + 1:  # 如果max_doc_len>13的话
            rel_mask = np.array(list(map(lambda x: -k <= x <= k, rel_pos.tolist())), dtype=np.int)
            rel_mask = torch.BoolTensor(rel_mask).to(DEVICE)
            rel_pos = rel_pos.masked_select(rel_mask)
            emo_pos = emo_pos.masked_select(rel_mask)
            cau_pos = cau_pos.masked_select(rel_mask)
            rel_mask = rel_mask.unsqueeze(1).expand(-1, 2 * feat_dim)
            rel_mask = rel_mask.unsqueeze(0).expand(batch, -1, -1)
            P = P.masked_select(rel_mask)
            P = P.reshape(batch, -1, 2 * feat_dim)

        emo_cau_pos = []
        for emo, cau in zip(emo_pos.tolist(), cau_pos.tolist()):
            emo_cau_pos.append([emo, cau])
        return P, emo_cau_pos  # 运行一下看看

    def loss_rank(self, couples_pred, emo_cau_pos, doc_couples, y_mask, test=False):
        couples_true, couples_mask, doc_couples_pred = self.output_util(couples_pred, emo_cau_pos, doc_couples, y_mask, test)

        if not self.pairwise_loss:
            couples_mask = torch.ByteTensor(couples_mask).to(DEVICE)
            couples_true = torch.FloatTensor(couples_true).to(DEVICE)
            criterion = nn.BCEWithLogitsLoss(reduction='mean')
            couples_true = couples_true.masked_select(couples_mask)
            couples_pred = couples_pred.masked_select(couples_mask)
            loss_couple = criterion(couples_pred, couples_true)
        else:
            x1, x2, y = self.pairwise_util(couples_pred, couples_true, couples_mask)
            criterion = nn.MarginRankingLoss(margin=1.0, reduction='mean')
            loss_couple = criterion(F.tanh(x1), F.tanh(x2), y)

        return loss_couple, doc_couples_pred

    def output_util(self, couples_pred, emo_cau_pos, doc_couples, y_mask, test=False):
        """
        TODO: combine this function to data_loader
        """
        batch, n_couple = couples_pred.size()

        couples_true, couples_mask = [], []
        doc_couples_pred = []
        for i in range(batch):
            y_mask_i = y_mask[i]
            max_doc_idx = sum(y_mask_i)

            doc_couples_i = doc_couples[i]
            couples_true_i = []
            couples_mask_i = []
            for couple_idx, emo_cau in enumerate(emo_cau_pos):
                if emo_cau[0] > max_doc_idx or emo_cau[1] > max_doc_idx:
                    couples_mask_i.append(0)  # 不用mask
                    couples_true_i.append(0)
                else:
                    couples_mask_i.append(1)
                    couples_true_i.append(1 if emo_cau in doc_couples_i else 0)

            couples_pred_i = couples_pred[i]
            doc_couples_pred_i = []
            if test:
                if torch.sum(torch.isnan(couples_pred_i)) > 0:
                    k_idx = [0] * 3
                else:
                    _, k_idx = torch.topk(couples_pred_i, k=3, dim=0)
                doc_couples_pred_i = [(emo_cau_pos[idx], couples_pred_i[idx].tolist()) for idx in k_idx]

            couples_true.append(couples_true_i)
            couples_mask.append(couples_mask_i)
            doc_couples_pred.append(doc_couples_pred_i)
        return couples_true, couples_mask, doc_couples_pred

    def loss_pre(self, pred_e, pred_c, y_emotions, y_causes, y_mask):
        y_mask = torch.ByteTensor(y_mask).to(DEVICE)
        y_emotions = torch.FloatTensor(y_emotions).to(DEVICE)
        y_causes = torch.FloatTensor(y_causes).to(DEVICE)

        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        pred_e = pred_e.masked_select(y_mask)
        true_e = y_emotions.masked_select(y_mask)
        pos_weight = torch.where(true_e == 1, 1.4, 1.0)
        # pos_weight = torch.where(true_e == 1, 1.5, 1.0)
        criterion1 = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
        loss_e = criterion(pred_e, true_e)

        pred_c = pred_c.masked_select(y_mask)
        true_c = y_causes.masked_select(y_mask)
        # pos_weight2 = torch.where(true_c == 1, 1.5, 1.0)

        # criterion2 = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight2)
        loss_c = criterion(pred_c, true_c)
        return loss_e, loss_c

    def pairwise_util(self, couples_pred, couples_true, couples_mask):
        """
        TODO: efficient re-implementation; combine this function to data_loader
        """
        batch, n_couple = couples_pred.size()
        x1, x2 = [], []
        for i in range(batch):
            x1_i_tmp = []
            x2_i_tmp = []
            couples_mask_i = couples_mask[i]
            couples_pred_i = couples_pred[i]
            couples_true_i = couples_true[i]
            for pred_ij, true_ij, mask_ij in zip(couples_pred_i, couples_true_i, couples_mask_i):
                if mask_ij == 1:
                    if true_ij == 1:
                        x1_i_tmp.append(pred_ij.reshape(-1, 1))
                    else:
                        x2_i_tmp.append(pred_ij.reshape(-1))
            m = len(x2_i_tmp)
            n = len(x1_i_tmp)
            x1_i = torch.cat([torch.cat(x1_i_tmp, dim=0)] * m, dim=1).reshape(-1)
            x1.append(x1_i)
            x2_i = []
            for _ in range(n):
                x2_i.extend(x2_i_tmp)
            x2_i = torch.cat(x2_i, dim=0)
            x2.append(x2_i)

        x1 = torch.cat(x1, dim=0)
        x2 = torch.cat(x2, dim=0)
        y = torch.FloatTensor([1] * x1.size(0)).to(DEVICE)
        return x1, x2, y

    def local_graph(self, doc_local_sents_h, doc_id, seq_len):
        # ============== LOCAL GRAPH ===============
        all_graphs = []
        all_span_infos = []
        all_node_features = []
        bs = len(doc_id)


        for d_id in range(bs):
            id = int(doc_id[d_id])

            graphs_file = os.path.join("../data/dgl_data", '%d.bin' % id)
            graph_list = load_graphs(graphs_file)  # glist will be [g1]

            span_info = []  # node_num * 2
            for g in graph_list[0]:
                all_graphs.append(g)
                span_info.append(g.ndata['span'])

            span_info = torch.cat(span_info, dim=0)
            node_num = span_info.size(0)
            all_span_infos.append(span_info)




            graph_span_mask = torch.arange(seq_len).unsqueeze(0).repeat(node_num, 1)
            graph_span_mask = (graph_span_mask >= span_info[:, 0:1]) & (graph_span_mask <= span_info[:, 1:])
            graph_span_mask = graph_span_mask.float()
            graph_span_mask_num = torch.sum(graph_span_mask, dim=-1, keepdim=True)
            graph_span_mask_num = (graph_span_mask_num == 0).float() + graph_span_mask_num
            node_feature = torch.mm(graph_span_mask.to(DEVICE),
                                    doc_local_sents_h[d_id].to(DEVICE)) / graph_span_mask_num.to(DEVICE)
            all_node_features.append(node_feature)
        node_features_big = torch.cat(all_node_features, dim=0)
        batched_graph = dgl.batch(all_graphs)

        feature_bank = [node_features_big]
        for GCN_layer in self.GCN_layers:
            node_features_big = GCN_layer(batched_graph, {"node": node_features_big})["node"]
            feature_bank.append(node_features_big)
        feature_bank = torch.cat(feature_bank, dim=-1)
        feature_bank = self.middle_layer(feature_bank)

        cur_bias = 0
        all_local_graph_feature = []
        for cur_span_info in all_span_infos:
            cur_node_num = cur_span_info.size(0)
            cur_features_bank = feature_bank[cur_bias:cur_bias + cur_node_num]  # node_num * hidden_size
            cur_bias += cur_node_num
            graph_span_mask = torch.arange(seq_len).unsqueeze(0).repeat(cur_node_num, 1)
            graph_span_mask = (graph_span_mask >= cur_span_info[:, 0:1]) & (graph_span_mask <= cur_span_info[:, 1:])
            graph_span_mask = graph_span_mask.t()  # seq_len * node_num
            graph_span_mask = graph_span_mask.float()
            graph_span_mask_num = torch.sum(graph_span_mask, dim=-1, keepdim=True)
            graph_span_mask_num = (graph_span_mask_num == 0).float() + graph_span_mask_num
            local_graph_feature = torch.mm(graph_span_mask.to(DEVICE),
                                           cur_features_bank.to(DEVICE)) / graph_span_mask_num.to(DEVICE)
            all_local_graph_feature.append(local_graph_feature.unsqueeze(0))

        final_local_graph_feature = torch.cat(all_local_graph_feature, dim=0)





        return final_local_graph_feature




class Pre_Predictions(nn.Module):
    def __init__(self, configs):
        super(Pre_Predictions, self).__init__()
        self.feat_dim = int(configs.gnn_dims.strip().split(',')[-1]) * int(configs.att_heads.strip().split(',')[-1])
        # 768

        self.out_e = nn.Linear(self.feat_dim, 1)  # (768,1)
        self.out_c = nn.Linear(self.feat_dim, 1)  # (768,1)

    def forward(self, doc_sents_h):
        pred_e = self.out_e(doc_sents_h)
        pred_c = self.out_c(doc_sents_h)
        return pred_e.squeeze(2), pred_c.squeeze(2)


class Couple(nn.Module):
    def __init__(self, configs):
        super(Couple, self).__init__()
        self.K = configs.K  # 12
        self.pos_emb_dim = configs.pos_emb_dim
        self.pos_layer = nn.Embedding(2 * self.K + 1, self.pos_emb_dim)  # (25,50),size:1
        nn.init.xavier_uniform_(self.pos_layer.weight)

        self.feat_dim = int(configs.gnn_dims.strip().split(',')[-1]) * int(configs.att_heads.strip().split(',')[-1])
        # 768
        self.rank_feat_dim = 2 * self.feat_dim + self.pos_emb_dim
        # 2*768+50 = 1586
        self.rank_layer1 = nn.Linear(self.rank_feat_dim, self.rank_feat_dim)  # (1586，1586)
        self.rank_layer2 = nn.Linear(self.rank_feat_dim, 1)  # (1586,1)


    def forward(self, doc_sents_h):
        batch, _, _ = doc_sents_h.size()  # 2,max_doc_len,768
        couples, rel_pos, emo_cau_pos = self.couple_generator(doc_sents_h, self.K)

        rel_pos = rel_pos + self.K
        rel_pos_emb = self.pos_layer(rel_pos)  # pos_layer:(25,50)

        couples = torch.cat([couples, rel_pos_emb], dim=2)
        couples = F.relu(self.rank_layer1(couples))  # (1586,1586)
        couples_pred = self.rank_layer2(couples)  # （1586，1）
        return couples_pred.squeeze(2), emo_cau_pos

    def couple_generator(self, H, k):  # H:doc_sents_h
        batch, seq_len, feat_dim = H.size()
        P_left = torch.cat([H] * seq_len, dim=2)
        P_left = P_left.reshape(-1, seq_len * seq_len, feat_dim)
        P_right = torch.cat([H] * seq_len, dim=1)
        P = torch.cat([P_left, P_right], dim=2)

        base_idx = np.arange(1, seq_len + 1)
        emo_pos = np.concatenate([base_idx.reshape(-1, 1)] * seq_len, axis=1).reshape(1, -1)[0]
        cau_pos = np.concatenate([base_idx] * seq_len, axis=0)

        rel_pos = cau_pos - emo_pos  # 相对位置
        rel_pos = torch.LongTensor(rel_pos).to(DEVICE)
        emo_pos = torch.LongTensor(emo_pos).to(DEVICE)
        cau_pos = torch.LongTensor(cau_pos).to(DEVICE)

        if seq_len > k + 1:
            rel_mask = np.array(list(map(lambda x: -k <= x <= k, rel_pos.tolist())), dtype=np.int)
            rel_mask = torch.BoolTensor(rel_mask).to(DEVICE)
            rel_pos = rel_pos.masked_select(rel_mask)
            emo_pos = emo_pos.masked_select(rel_mask)
            cau_pos = cau_pos.masked_select(rel_mask)
            rel_mask = rel_mask.unsqueeze(1).expand(-1, 2 * feat_dim)
            rel_mask = rel_mask.unsqueeze(0).expand(batch, -1, -1)
            P = P.masked_select(rel_mask)
            P = P.reshape(batch, -1, 2 * feat_dim)

        assert rel_pos.size(0) == P.size(1)
        rel_pos = rel_pos.unsqueeze(0).expand(batch, -1)

        emo_cau_pos = []
        for emo, cau in zip(emo_pos.tolist(), cau_pos.tolist()):
            emo_cau_pos.append([emo, cau])
        return P, rel_pos, emo_cau_pos


class RankNN(nn.Module):
    def __init__(self, configs):
        super(RankNN, self).__init__()
        self.K = configs.K  # 12
        self.pos_emb_dim = configs.pos_emb_dim
        self.pos_layer = nn.Embedding(2 * self.K + 1, self.pos_emb_dim)
        nn.init.xavier_uniform_(self.pos_layer.weight)

        self.feat_dim = int(configs.gnn_dims.strip().split(',')[-1]) * int(configs.att_heads.strip().split(',')[-1])
        self.rank_feat_dim = 2 * self.feat_dim + self.pos_emb_dim

        self.rank_layer1 = nn.Linear(self.rank_feat_dim, self.rank_feat_dim)
        self.rank_layer2 = nn.Linear(self.rank_feat_dim, 1)

    def forward(self, doc_sents_h):
        batch, max_doc_len, feat_dim = doc_sents_h.size()
        couples, rel_pos, emo_cau_pos = self.couple_generator(doc_sents_h, self.K)

        rel_pos = rel_pos + self.K
        rel_pos_emb = self.pos_layer(rel_pos)
        kernel = self.kernel_generator(rel_pos)
        kernel = kernel.unsqueeze(0).expand(batch, -1, -1)
        rel_pos_emb = torch.matmul(kernel, rel_pos_emb)
        couples = torch.cat([couples, rel_pos_emb], dim=2)

        couples = F.relu(self.rank_layer1(couples))
        couples_pred = self.rank_layer2(couples)
        return couples_pred.squeeze(2), emo_cau_pos, couples

    def couple_generator(self, H, k):
        batch, seq_len, feat_dim = H.size()
        P_left = torch.cat([H] * seq_len, dim=2)
        P_left = P_left.reshape(-1, seq_len * seq_len, feat_dim)
        P_right = torch.cat([H] * seq_len, dim=1)
        P = torch.cat([P_left, P_right], dim=2)

        base_idx = np.arange(1, seq_len + 1)
        emo_pos = np.concatenate([base_idx.reshape(-1, 1)] * seq_len, axis=1).reshape(1, -1)[0]
        cau_pos = np.concatenate([base_idx] * seq_len, axis=0)

        rel_pos = cau_pos - emo_pos  # 相对位置
        rel_pos = torch.LongTensor(rel_pos).to(DEVICE)
        emo_pos = torch.LongTensor(emo_pos).to(DEVICE)
        cau_pos = torch.LongTensor(cau_pos).to(DEVICE)

        if seq_len > k + 1:
            rel_mask = np.array(list(map(lambda x: -k <= x <= k, rel_pos.tolist())), dtype=np.int)
            rel_mask = torch.BoolTensor(rel_mask).to(DEVICE)
            rel_pos = rel_pos.masked_select(rel_mask)
            emo_pos = emo_pos.masked_select(rel_mask)
            cau_pos = cau_pos.masked_select(rel_mask)
            rel_mask = rel_mask.unsqueeze(1).expand(-1, 2 * feat_dim)
            rel_mask = rel_mask.unsqueeze(0).expand(batch, -1, -1)
            P = P.masked_select(rel_mask)
            P = P.reshape(batch, -1, 2 * feat_dim)

        assert rel_pos.size(0) == P.size(1)
        rel_pos = rel_pos.unsqueeze(0).expand(batch, -1)

        emo_cau_pos = []
        for emo, cau in zip(emo_pos.tolist(), cau_pos.tolist()):
            emo_cau_pos.append([emo, cau])
        return P, rel_pos, emo_cau_pos

    def kernel_generator(self, rel_pos):
        n_couple = rel_pos.size(1)
        rel_pos_ = rel_pos[0].type(torch.FloatTensor).to(DEVICE)
        kernel_left = torch.cat([rel_pos_.reshape(-1, 1)] * n_couple, dim=1)
        kernel = kernel_left - kernel_left.transpose(0, 1)
        return torch.exp(-(torch.pow(kernel, 2)))


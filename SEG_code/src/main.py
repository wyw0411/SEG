import sys, os, warnings

import torch

sys.path.append('..')
warnings.filterwarnings("ignore")
from data_loader import *
from rank_cp import *
from transformers import AdamW, get_linear_schedule_with_warmup
from src.utils.utils import *
import time
from transformers import logging
result_dir = './'
logfile = os.path.join(result_dir, 'log.txt')
logging.set_verbosity_warning()
logging.set_verbosity_error()  # 无视出现的一些警告和错误

def main(configs, fold_id):
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True  # 设置求导的算法是固定的

    train_loader = build_train_data(configs, fold_id=fold_id)
    if configs.split == 'split20':
        valid_loader = build_inference_data(configs, fold_id=fold_id, data_type='valid')
    test_loader = build_inference_data(configs, fold_id=fold_id, data_type='test')
    # 验证集和测试集数据不用打乱
    model = Network(configs).to(DEVICE)

    params = model.parameters()
    params_bert = model.bert.parameters()
    params_rest = list(model.pred.parameters()) + list(model.gcn.parameters()) + list(model.middle_layer.parameters()) +\
        list(model.rank.parameters()) + list(model.emo_class.parameters())
    assert sum([param.nelement() for param in params]) == \
           sum([param.nelement() for param in params_bert]) + sum([param.nelement() for param in params_rest])


    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': configs.l2_bert, 'eps': configs.adam_epsilon},
        {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'eps': configs.adam_epsilon},
        {'params': params_rest,
         'weight_decay': configs.l2}
    ]

    optimizer = AdamW(params, lr=configs.lr)  # adam是深度学习中用来替代随即下降的优化算法，

    num_steps_all = len(train_loader) // configs.gradient_accumulation_steps * configs.epochs
    warmup_steps = int(num_steps_all * configs.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps_all)

    model.zero_grad()  # 梯度置零
    max_ec, max_e, max_c = (-1, -1, -1), None, None
    metric_ec, metric_e, metric_c = (-1, -1, -1), None, None
    early_stop_flag = None
    for epoch in range(1, configs.epochs+1):
        m = 0
        for train_step, batch in enumerate(train_loader, 1):
            model.train()
            doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
            bert_token_b, bert_segment_b, bert_global_masks_b, bert_local_masks_b, bert_local_token_b, bert_local_segment_b, bert_clause_b, bert_sep_b, doc_str, emotion_type = batch

            x = np.zeros((bert_clause_b.size(0), bert_clause_b.size(1)))
            for i in range(bert_clause_b.size(0)):
                for j in range(bert_clause_b.size(1)):
                    if j == 0:
                        x[i][0] = 1
                    elif bert_clause_b[i][j] > 0:
                        x[i][j] = 1
                    else:
                        x[i][j] = 0
            x = torch.tensor(x)
            mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

            couples_pred, emo_cau_pos, pred_e, pred_c = model(bert_token_b, bert_segment_b, bert_global_masks_b, bert_local_masks_b, bert_local_token_b, bert_local_segment_b,
                                                              bert_clause_b, bert_sep_b, doc_len_b, adj_b, fold_id, epoch, doc_id_b, doc_str, mask,emotion_type, y_emotions_b, doc_couples_b)


            loss_e, loss_c = model.loss_pre(pred_e, pred_c, y_emotions_b, y_causes_b, y_mask_b)
            loss_couple, _ = model.loss_rank(couples_pred, emo_cau_pos, doc_couples_b, y_mask_b, test=False)
            loss = loss_couple + loss_e + loss_c

            loss = loss / configs.gradient_accumulation_steps
            if m % 50 == 0:
                print("epoch:{} :{},  loss_cp:{},  loss_e:{}, loss_c:{}".format(epoch, m, loss_couple, loss_e, loss_c))
            m += 1

            loss.backward()
            if train_step % configs.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()



        with torch.no_grad():
            model.eval()

            if configs.split == 'split10':
                test_ec, test_e, test_c, doc_id_all, doc_couples_all, doc_couples_pred_all = inference_one_epoch(configs, test_loader, model, epoch)
                if test_ec[2] > metric_ec[2]:
                    early_stop_flag = 1
                    metric_ec, metric_e, metric_c = test_ec, test_e, test_c  # 因果抽取，情感抽取，原因抽取
                else:
                    early_stop_flag += 1

            if configs.split == 'split20':
                valid_ec, valid_e, valid_c, _, _, _ = inference_one_epoch(configs, valid_loader, model)
                test_ec, test_e, test_c, _, _, _ = inference_one_epoch(configs, test_loader, model)
                if valid_ec[2] > max_ec[2]:
                    early_stop_flag = 1
                    max_ec, max_e, max_c = valid_ec, valid_e, valid_c
                    metric_ec, metric_e, metric_c = test_ec, test_e, test_c
                else:
                    early_stop_flag += 1

        if epoch > configs.epochs / 2 and early_stop_flag >= 5:
            break

    return metric_ec, metric_e, metric_c




def inference_one_batch(configs, batch, model, epoch):
    doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
    bert_token_b, bert_segment_b, bert_global_masks_b, bert_local_masks_b, bert_local_token_b, bert_local_segment_b, bert_clause_b, bert_sep_b, doc_str, emotion_type = batch
    x = np.zeros((bert_clause_b.size(0), bert_clause_b.size(1)))
    for i in range(bert_clause_b.size(0)):
        for j in range(bert_clause_b.size(1)):
            if j == 0:
                x[i][0] = 1
            elif bert_clause_b[i][j] > 0:
                x[i][j] = 1
            else:
                x[i][j] = 0
    x = torch.tensor(x)
    mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

    couples_pred, emo_cau_pos, pred_e, pred_c = model(bert_token_b, bert_segment_b, bert_global_masks_b,
                                                      bert_local_masks_b, bert_local_token_b, bert_local_segment_b,
                                                      bert_clause_b, bert_sep_b, doc_len_b, adj_b, fold_id, epoch,
                                                      doc_id_b, doc_str, mask, emotion_type, y_emotions_b, doc_couples_b)

    loss_e, loss_c = model.loss_pre(pred_e, pred_c, y_emotions_b, y_causes_b, y_mask_b)
    loss_couple, doc_couples_pred_b = model.loss_rank(couples_pred, emo_cau_pos, doc_couples_b, y_mask_b, test=True)



    return to_np(loss_couple), to_np(loss_e), to_np(loss_c),\
           doc_couples_b, doc_couples_pred_b, doc_id_b


def inference_one_epoch(configs, batches, model, epoch):
    doc_id_all, doc_couples_all, doc_couples_pred_all = [], [], []
    for batch in batches:
        _, _, _, doc_couples, doc_couples_pred, doc_id_b = inference_one_batch(configs, batch, model, epoch)
        # print("doc_id:{},doc_pred:{}".format(doc_id_b,doc_couples_pred))
        doc_id_all.extend(doc_id_b)
        doc_couples_all.extend(doc_couples)
        doc_couples_pred_all.extend(doc_couples_pred)

    doc_couples_pred_all = lexicon_based_extraction(doc_id_all, doc_couples_pred_all)



    metric_ec, metric_e, metric_c = eval_func(doc_couples_all, doc_couples_pred_all)  # 分别求因果抽取，情感抽取和原因抽取的F，P,R
    return metric_ec, metric_e, metric_c, doc_id_all, doc_couples_all, doc_couples_pred_all

def pair_extraction(doc_ids, couples_pred, doc_couples_all, threshold=0.5):
    couples_pred_filtered = []
    for i, (doc_id, couples_pred_i, doc_couple) in enumerate(zip(doc_ids, couples_pred, doc_couples_all)):
        top1, top1_prob = couples_pred_i[0][0], couples_pred_i[0][1]
        couples_pred_i_filtered = [top1]
        for couple in couples_pred_i[1:]:
            if logistic(couple[1]) > threshold:
                couples_pred_i_filtered.append(couple[0])
        couples_pred_filtered.append(couples_pred_i_filtered)
    return couples_pred_filtered

def lexicon_based_extraction(doc_ids, couples_pred):  # 取top1，然后根据后面的是否包含字典里的情感来决定是否有因果关系
    emotional_clauses = read_b(os.path.join(DATA_DIR, SENTIMENTAL_CLAUSE_DICT))

    couples_pred_filtered = []
    for i, (doc_id, couples_pred_i) in enumerate(zip(doc_ids, couples_pred)):
        if couples_pred_i != []:
            top1, top1_prob = couples_pred_i[0][0], couples_pred_i[0][1]
            # print(top1)
            # print(type([top1]))
            couples_pred_i_filtered = [top1]
            # print(couples_pred_i_filtered)
            emotional_clauses_i = emotional_clauses[doc_id]
            for couple in couples_pred_i[1:]:
                if couple[0][0] in emotional_clauses_i and logistic(couple[1]) > 0.5:
                    couples_pred_i_filtered.append(couple[0])  # 如果剩余的候选对情感包含在lexion中，那么跟这个情感有关的候选对都加入
        else:
            couples_pred_i_filtered = [[1, 1]]

        couples_pred_filtered.append(couples_pred_i_filtered)
    return couples_pred_filtered


if __name__ == '__main__':
    configs = Config()
    start_time = time.time()


    if configs.split == 'split10':  # 做十次交叉验证
        n_folds = 10
        configs.epochs = 20
    elif configs.split == 'split20':
        n_folds = 20
        configs.epochs = 15
    else:
        print('Unknown data split.')
        exit()

    metric_folds = {'ecp': [], 'emo': [], 'cau': []}
    for fold_id in range(1, n_folds+1):  # 1到10
        print('===== fold {} ====='.format(fold_id))
        metric_ec, metric_e, metric_c = main(configs, fold_id)
        print('F_ecp: {}'.format(float_n(metric_ec[2])))

        metric_folds['ecp'].append(metric_ec)
        metric_folds['emo'].append(metric_e)
        metric_folds['cau'].append(metric_c)

    metric_ec = np.mean(np.array(metric_folds['ecp']), axis=0).tolist()
    metric_e = np.mean(np.array(metric_folds['emo']), axis=0).tolist()
    metric_c = np.mean(np.array(metric_folds['cau']), axis=0).tolist()


    print('===== Average =====')
    print('F_ecp: {}, P_ecp: {}, R_ecp: {}'.format(float_n(metric_ec[2]), float_n(metric_ec[0]), float_n(metric_ec[1])))
    print('F_emo: {}, P_emo: {}, R_emo: {}'.format(float_n(metric_e[2]), float_n(metric_e[0]), float_n(metric_e[1])))
    print('F_cau: {}, P_cau: {}, R_cau: {}'.format(float_n(metric_c[2]), float_n(metric_c[0]), float_n(metric_c[1])))
    print(f'Total running time:{(time.time() - start_time) / 3600}hours')
    write_b({'ecp': metric_ec, 'emo': metric_e, 'cau': metric_c, 'running time': (time.time()-start_time)/3600}, '1_split10_metrics.pkl')







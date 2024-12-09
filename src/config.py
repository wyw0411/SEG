import torch
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 333
DATA_DIR = '../data'
TRAIN_FILE = 'fold%s_train.json'
VALID_FILE = 'fold%s_valid.json'
TEST_FILE = 'fold%s_test.json'
MULTI_TEST_FILE = 'fold%s_test_multi.json'
SENTIMENTAL_CLAUSE_DICT = 'sentimental_clauses.pkl'


class Config(object):
    def __init__(self):
        self.n_tasks = 3
        self.split = 'split10'
        self.num_class = 6

        self.N = 1
        self.tran_dims = '192'
        self.att_heads = 1
        self.d_model = 768
        self.hidden = 3072

        self.bert_cache_path = 'bert-base-chinese'
        self.feat_dim = 768

        self.gnn_dims = '192'
        self.att_heads = '4'
        self.K = 2
        self.pos_emb_dim = 50
        self.pairwise_loss = False

        self.epochs = 50
        self.lr = 3e-5
        self.batch_size = 4
        self.gradient_accumulation_steps = 2
        self.dp = 0.1
        self.l2 = 1e-5
        self.l2_bert = 0.013
        self.warmup_proportion = 0.1
        self.adam_epsilon = 1e-8


import collections
import re
from transformers import BertTokenizer
from functools import partial
import jieba
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import torch.nn.functional as F
from transformers import AdamW

class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = partial(jieba.cut, HMM=False)

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


class KeyDataset(Dataset):
    def __init__(self, dict_data):
        self.data = dict_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class EncoderDecoderData:
    def __init__(self, args, tokenizer, ):
        self.train_data = self.read_file(args.train_file) if args.train_file else None
        self.dev_data = self.read_file(args.dev_file) if args.dev_file else None
        self.predict_data = self.read_file(args.predict_file) if args.predict_file else None
        self.args = args
        self.tokenizer = tokenizer

    def read_file(self, file):
        datalist = []
        data = ''
        for i in open(file, 'r', encoding='utf-8').readlines():
            if i != '\n':
                i = i.replace('\n', '[SEP]')  # 使用[SEP]表示段落结束
                data += i
            else:
                datalist.append(data)
                data = ''
                continue
        return datalist

    def train_collate(self, batch):
        res = self.tokenizer(batch,
                             padding=True,
                             return_tensors='pt',
                             max_length=512,
                             truncation='longest_first',
                             return_attention_mask=True,
                             return_token_type_ids=False)
        # res['decoder_attention_mask'] = res['attention_mask']
        res['labels'] = res['input_ids'].clone()
        res['decoder_input_ids'] = res['input_ids'].clone()
        masked_indices = torch.bernoulli(torch.full(res['labels'].shape, self.args.mlm_probability)).bool()
        res['decoder_input_ids'][masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        res['labels'][~masked_indices] = -1  # 85%原始数据没有被mask的位置进行赋值为-1

        if self.args.noise_prob:  # 选取noise_pro的token随机替换为随机值
            ids = res['input_ids'].clone()
            mask = res['decoder_attention_mask']
            noise_ids = torch.randint_like(ids, 1, 50000)
            noise_place = np.random.random(ids.shape) < self.args.noise_prob
            noise_place = torch.from_numpy(noise_place) & mask.bool()
            ids = torch.where(noise_place, noise_ids, ids)
            res['decoder_input_ids'] = ids
        return res

    def dev_collate(self, batch):
        return self.train_collate(batch)


def create_optimizer(model, lr, weight_decay, custom_lr=None):
    no_decay = 'bias|norm'
    params = collections.defaultdict(list)
    custom_lr = custom_lr or dict()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        in_custom = False
        for custom_name, _ in custom_lr.items():
            if custom_name in name:
                if re.search(no_decay, name.lower()):
                    params[custom_name].append(param)
                else:
                    params[custom_name + '_decay'].append(param)
                in_custom = True
                break
        if not in_custom:
            if re.search(no_decay, name.lower()):
                params['normal'].append(param)
            else:
                params['normal_decay'].append(param)

    optimizer_grouped_parameters = []
    for k, v in params.items():
        param_lr = custom_lr.get(k.split('_')[0], lr)
        decay = weight_decay if 'decay' in k else 0.0
        optimizer_grouped_parameters.append({'params':v, 'weight_decay': decay, 'lr': param_lr})

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer
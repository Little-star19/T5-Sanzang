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
import logging
import random

jieba.setLogLevel(logging.INFO)

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
        self.args = args
        self.tokenizer = tokenizer
        self.train_data = self.read_file(args.train_file) if args.train_file else None
        self.dev_data = self.read_file(args.dev_file) if args.dev_file else None
        self.predict_data = self.read_file(args.predict_file) if args.predict_file else None


    def read_file(self, file):
        datalist = []
        data = ''
        # 读取数据
        for i in open(file, 'r', encoding='utf-8').readlines():
            if i != '\n':
                i = i.replace('\n', '')
                data += i
            else:
                datalist.append(data)
                data = ''
                continue
        # MASK标记
        # masklist: ' <extra_id_0> walks in <extra_id_1> park '
        # targetlist: ' <extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s> '
        mask_dict = []
        for data in datalist:
            targetlist = []
            mask_dict_i = {'masklist': '', 'targetlist': ''}
            mask_num = int(float(len(data)) * 0.15)
            data_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(data))
            mask_index = random.sample(range(len(data_ids)), mask_num)
            for i in range(mask_num):  # 使用[unusedX]进行mask，masked token添加到目标序列
                targetlist.append('[unused' + str(i+1) + ']' + self.tokenizer.convert_ids_to_tokens(data_ids[mask_index[i]]))
                data_ids[mask_index[i]] = i+1
            targetlist = ''.join(targetlist)
            masklist = ''.join(self.tokenizer.convert_ids_to_tokens(data_ids))
            mask_dict_i['targetlist'] = targetlist
            mask_dict_i['masklist'] = masklist
            mask_dict.append(mask_dict_i)
        return mask_dict

    def train_collate(self, batch):
        # input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
        # labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>',return_tensors='pt').input_ids
        source = [x['targetlist'] for x in batch]
        target = [x['masklist'] for x in batch]
        res = self.tokenizer(source,
                             padding=True,
                             return_tensors='pt',
                             max_length=512,
                             truncation='longest_first',
                             return_attention_mask=True,
                             return_token_type_ids=False)
        target_features = self.tokenizer(target,
                                         padding=True,
                                         return_tensors='pt',
                                         max_length=150,
                                         truncation='longest_first',
                                         return_attention_mask=True,
                                         return_token_type_ids=False)
        res['labels'] = target_features['input_ids']
        return res

    def dev_collate(self, batch):
        return self.train_collate(batch)


    def get_dataloader(self):
        ret = {'train': [], 'dev': []}
        base_dataset = KeyDataset(self.train_data)
        if self.args.kfold > 1:
            from sklearn.model_selection import KFold    # 交叉验证
            for train_idx, dev_idx in KFold(n_splits=self.args.kfold, shuffle=True,
                                            random_state=self.args.seed).split(range(len(self.train_data))):
                train_dataset = Subset(base_dataset, train_idx)
                dev_dataset = Subset(base_dataset, dev_idx)
                train_dataloader = DataLoader(train_dataset,
                                              batch_size=self.args.batch_size,
                                              collate_fn=self.train_collate,
                                              num_workers=self.args.num_works,
                                              shuffle=True)
                dev_dataloader = DataLoader(dev_dataset,
                                            batch_size=self.args.batch_size * 2,
                                            collate_fn=self.dev_collate
                                            )
                ret['train'].append(train_dataloader)
                ret['dev'].append(dev_dataloader)
        else:
            if self.args.kfold == 1:
                from sklearn.model_selection import train_test_split
                train_idx, dev_idx = train_test_split(range(len(self.train_data)),
                                                      train_size=0.8,
                                                      test_size=0.2,
                                                      random_state=self.args.seed)
                train_dataset = Subset(base_dataset, train_idx)
                dev_dataset = Subset(base_dataset, dev_idx)
            else:
                assert self.dev_data is not None, 'When no kfold, dev data must be targeted'
                train_dataset = base_dataset
                dev_dataset = KeyDataset(self.dev_data)

            train_dataloader = DataLoader(train_dataset,
                                          batch_size=self.args.batch_size,
                                          collate_fn=self.train_collate,
                                          num_workers=self.args.num_works, shuffle=True)
            # for inputs in train_dataloader:
            #     print(inputs)
            dev_dataloader = DataLoader(dev_dataset,
                                        batch_size=self.args.batch_size * 2,
                                        collate_fn=self.dev_collate
                                        )
            ret['train'].append(train_dataloader)
            ret['dev'].append(dev_dataloader)
        return ret

def mask_select(inputs, mask):
    input_dim = inputs.ndim
    mask_dim = mask.ndim
    mask = mask.reshape(-1).bool()
    if input_dim > mask_dim:
        inputs = inputs.reshape((int(mask.size(-1)),-1))[mask]
    else:
        inputs = inputs.reshape(-1)[mask]
    return inputs


def ce_loss(inputs, targets, mask):
    mask = mask[:, 1:]
    inputs = inputs[:, :-1]
    targets = targets[:, 1:]
    inputs = mask_select(inputs, mask)
    targets = mask_select(targets, mask)
    print(inputs,targets)
    loss = F.cross_entropy(inputs, targets)
    return loss

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
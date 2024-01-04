# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:26:02 2020

@author: DrLC
"""
import subprocess

import numpy as np
from scipy.special import kl_div
from sklearn.feature_extraction.text import CountVectorizer
from dataset import OJ104, remove_tail_padding, CodeChef
from lstm_classifier import LSTMClassifier, LSTMEncoder, GRUEncoder, GRUClassifier
from modifier import TokenModifier, InsModifier, token_to_code
from lstm_eval import evaluate, gettensor, write_gen_data_time
import torch
import argparse
import pickle, gzip
import os, sys, time, copy
import random
import numpy
import os
import build_dataset as bd
from sklearn.metrics import mutual_info_score


def select_min_info(new_x, x):
    min_mutual_info = float('inf')
    min_mutual_info_x = x[0]
    min_index = 0
    for index, _x in enumerate(new_x):
        mutual_info = mutual_info_score(x[0], _x)
        if mutual_info < min_mutual_info:
            min_mutual_info = mutual_info
            min_mutual_info_x = _x
            min_index = index
    return min_index, min_mutual_info_x




class SafeCodeToken(object):
    def __init__(self, dataset, symtab, classifier, device):

        self.tokenM = TokenModifier(classifier=classifier,
                                    loss=torch.nn.CrossEntropyLoss(),
                                    uids=symtab['all'],
                                    txt2idx=dataset.get_txt2idx(),
                                    idx2txt=dataset.get_idx2txt(),
                                    device=device)
        self.cl = classifier
        self.d = dataset
        self.syms = symtab
        self.device = device

    def attack(self, x, y, uids, n_candidate=100, n_iter=20):
        iter = 0
        n_stop = 0
        stop_iter = random.randint(3, n_iter)
        ori_x = copy.deepcopy(x)
        while iter < n_iter:
            keys = list(uids.keys())
            for k in keys:
                if n_stop >= len(uids):
                    iter = n_iter
                    return True, x, 1
                if k in self.tokenM.forbidden_uid:
                    n_stop += 1
                    continue
                if opt.do_noise:
                    new_x, new_uid_cand = self.tokenM.rename_uid_mu(x, y, uids[k], k, n_candidate,
                                                                    noise_scale=opt.index)
                else:
                    new_x, new_uid_cand = self.tokenM.rename_uid(x, y, uids[k], k, n_candidate)
                iter += 1
                if new_x is None:
                    n_stop += 1
                    continue
                if iter == stop_iter:
                    new_prob_idx = random.randint(0, len(new_x) - 1)
                    x = [new_x[new_prob_idx]]
                    return True, x, 1
                new_prob_idx = random.randint(0, len(new_x) - 1)
                x = [new_x[new_prob_idx]]
                uids[self.d.idx2vocab(int(new_uid_cand[new_prob_idx]))] = uids.pop(k)
        return False, x, 2

    def gen_all(self, n_candidate=100, n_iter=20, res_save=None, origin_data_path=None):
        st_time = time.time()
        with gzip.open(origin_data_path, "rb") as f:
            dd = pickle.load(f)
        print("Start generating samples...")
        total_samples = self.d.train.get_size()
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('simple all', total_samples)
        for i in range(total_samples):
            b = self.d.train.next_batch(1)
            tag, x, typ = self.attack(b['x'], b['y'], self.syms['tr'][b['id'][0]], n_candidate, n_iter)
            dd['x_tr'][b['id'][0]] = x[0]
            if (i + 1) % 500 == 0:  # Every x samples to print a schedule
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                print(f' {i + 1} done {total_samples} all.')
                print('The current progress：%.2f%%' % ((i + 1) / total_samples * 100))
                print('time consuming', time.time() - st_time)
        if res_save is not None:
            write_gen_data_time(res_save, st_time, self.d.train.get_size())
            with gzip.open(res_save, "wb") as f:
                pickle.dump(dd, f)
                print("success save file", res_save)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print("[Task Done] Time Cost: %.1f " % (time.time() - st_time))


class RandomToken(object):
    def __init__(self, dataset, symtab, classifier, device):

        self.tokenM = TokenModifier(classifier=classifier,
                                    loss=torch.nn.CrossEntropyLoss(),
                                    uids=symtab['all'],
                                    txt2idx=dataset.get_txt2idx(),
                                    idx2txt=dataset.get_idx2txt(),
                                    device=device)
        self.cl = classifier
        self.d = dataset
        self.syms = symtab
        self.device = device

    def attack(self, x, y, uids, n_candidate=100, n_iter=20):
        iter = 0
        n_stop = 0
        while iter < n_iter:
            keys = list(uids.keys())
            for k in keys:
                if n_stop >= len(uids):
                    iter = n_iter
                    return True, x, 1
                if k in self.tokenM.forbidden_uid:
                    n_stop += 1
                    continue
                new_x, new_uid_cand = self.tokenM.rename_uid_random(x, uids[k], k)
                iter += 1
                if new_x is None:
                    n_stop += 1
                    continue
                new_prob_idx = random.randint(0, len(new_x) - 1)
                x = new_x[new_prob_idx]
                return True, [x], 1

        return False, x, 2

    def gen_all(self, n_candidate=100, n_iter=20, res_save=None):
        st_time = time.time()
        with gzip.open(opt.data, "rb") as f:
            dd = pickle.load(f)
        print("Start generating samples...")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('To generate a of the total number of the samples：', self.d.train.get_size())
        for i in range(self.d.train.get_size()):
            b = self.d.train.next_batch(1)
            tag, x, typ = self.attack(b['x'], b['y'], self.syms['tr'][b['id'][0]], n_candidate, n_iter)
            dd['x_tr'][b['id'][0]] = x[0]

        if res_save is not None:
            write_gen_data_time(res_save, st_time, self.d.train.get_size())
            with gzip.open(res_save, "wb") as f:
                pickle.dump(dd, f)
                print("success save file")
        print("[Task Done] Time Cost: %.1f " % (time.time() - st_time))


class RandomDeadCode(object):

    def __init__(self, dataset, instab, classifier):

        self.insM = InsModifier(
            classifier=classifier,
            txt2idx=dataset.get_txt2idx(),
            poses=None)  # wait to init when attack
        self.d = dataset
        self.inss = instab
        self.cl = classifier

    def attack(self, x, y, poses, n_candidate=100, n_iter=20):
        self.insM.initInsertDict(poses)
        iter = 0
        n_stop = 0

        while iter < n_iter:
            iter += 1
            new_x, new_insertDict = self.insM.insert_remove_random(x[0])
            if not new_x:
                n_stop += 1
                continue
            # A return in new_x random sampling

            new_prob_idx = random.randint(0, len(new_x) - 1)
            x = new_x[new_prob_idx]
            x = x[:self.cl.max_len]
            return True, [x], 1

        return False, x, 2

    def gen_all(self, n_candidate=100, n_iter=20, res_save=None):
        st_time = time.time()
        with gzip.open(opt.data, "rb") as f:
            dd = pickle.load(f)
        for i in range(self.d.train.get_size()):
            b = self.d.train.next_batch(1)
            tag, x, typ = self.attack(b['x'], b['y'], self.inss['stmt_tr'][b['id'][0]], n_candidate, n_iter)
            dd['x_tr'][b['id'][0]] = x[0]

        if res_save is not None:
            write_gen_data_time(res_save, st_time, self.d.train.get_size())
            with gzip.open(res_save, "wb") as f:
                pickle.dump(dd, f)
                print("success save file")
        print("[Task Done] Time Cost: %.1f " % (time.time() - st_time))


"""
    conda activate lstm  && cd /home/yjy/code/2023/lstm-yjy
    nohup python -u attacker4simple.py --gpu 0  > 4simple_safecode_.txt 2>&1 &
"""
if __name__ == "__main__":
    # 0.The basic parameters
    root = '../data/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default="0")
    parser.add_argument('--attn', action='store_true')
    parser.add_argument('--enhance_method', default='safecode', help='safecode or carrot alert random')
    parser.add_argument('--model_name', default='LSTM', help='LSTM or BGRU')
    parser.add_argument('--task', default='code_defect', help='code_function or code_defect')
    parser.add_argument('--attack_type', default='token', help='dead_code , token')
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--do_noise', action='store_true', default=False)

    # safecode Sweep
    parser.add_argument('--wandb_name', type=str, default='default', help='The name of the wandb,Naming rules：The experiment purpose，QA')
    parser.add_argument('--iter_select', type=int, default=30, help='The number of iterations:5 10 20')
    parser.add_argument('--index', type=int, default=1, help='Expansion Angle of the:0 1 2 3 4')
    parser.add_argument('--candidate', type=int, default=50, help='candidate:100 30 10 50')
    parser.add_argument('--model_epoch', type=int, default=50, help='7,14,21')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device("cuda:" + opt.gpu)
    embedding_size = 512
    hidden_size = 600
    n_layers = 2
    domain = 'origin'
    opt.data = root + opt.task + '/dataset/origin/data.pkl.gz'

    if opt.task == 'code_function':
        max_len = 500
        n_class = 104
        vocab_size = 5000
        original_dataset = OJ104(path=opt.data)
        training_s = original_dataset.train
        valid_s = original_dataset.dev
        test_s = original_dataset.test
        poj = OJ104(path=root + opt.task + "/dataset/origin/data.pkl.gz",
                    max_len=max_len,
                    vocab_size=vocab_size)
        training_set = poj.train
        valid_set = poj.dev
        test_set = poj.test

    elif opt.task == 'code_defect':
        n_class = 4
        vocab_size = 3000
        max_len = 300
        original_dataset = CodeChef(path=opt.data)
        training_s = original_dataset.train
        valid_s = original_dataset.dev
        test_s = original_dataset.test
        poj = CodeChef(path=root + opt.task + "/dataset/origin/data.pkl.gz",
                       max_len=max_len,
                       vocab_size=vocab_size)
        training_set = poj.train
        valid_set = poj.dev
        test_set = poj.test

    enc = LSTMEncoder(embedding_size, hidden_size, n_layers)
    classifier = LSTMClassifier(vocab_size, embedding_size, enc,
                                hidden_size, n_class, max_len, attn=opt.attn).to(device)

    if opt.enhance_method == 'safecode' or opt.enhance_method == 'alert':
        if opt.task == 'code_defect' and opt.attack_type == 'token':
            model_path = os.path.join(root, opt.task, "model", opt.model_name,
                                      "safecode/token", opt.wandb_name, str(opt.model_epoch) + ".pt")
        else:
            model_path = os.path.join(root, opt.task, "model", opt.model_name, 'origin', "best.pt")

        classifier.load_state_dict(torch.load(model_path))

    # Expand the sample name
    if opt.enhance_method == 'safecode':
        opt.data_name = opt.enhance_method + '-' + opt.attack_type
        if opt.task == 'code_defect' and opt.attack_type == 'token':
            # opt.data_name = opt.data_name + '-iter_select' + str(opt.iter_select) + '-index-' + str(opt.index)
            opt.data_name = opt.data_name + '-' + opt.wandb_name + '-' + str(opt.index)
        elif opt.attack_type == 'dead_code':
            opt.data_name = opt.data_name + '-' + str(opt.index)
    else:
        opt.data_name = opt.enhance_method + '-' + opt.attack_type
    save_path = os.path.join(root, opt.task, "dataset", opt.model_name, opt.data_name)
    print('data save at :', save_path)
    os.makedirs(save_path, exist_ok=True)

    # GPU choose
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device("cuda:" + opt.gpu)

    # Word map
    with gzip.open(root + opt.task + "/dataset/origin/data.pkl.gz", "rb") as f:
        dd = pickle.load(f)
    with gzip.open(root + opt.task + '/dataset/origin/data_uid.pkl.gz', "rb") as f:  # 'all'
        symtab = pickle.load(f)
    with gzip.open(root + opt.task + '/dataset/origin/data_inspos.pkl.gz', "rb") as f:
        instab = pickle.load(f)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # Dead code generated against attacks
    if opt.enhance_method == 'alert':
        if opt.attack_type == 'dead_code':
            atk = AlertDeadCode(poj, instab, classifier)
            atk.attack_all(40, 20, res_save=save_path + "/data.pkl.gz")
        elif opt.attack_type == 'token':
            atk = AlertToken(poj, symtab, classifier, device)
            atk.attack_all(40, 50, res_save=save_path + "/data.pkl.gz")
    elif opt.enhance_method == 'safecode':
        if opt.attack_type == 'dead_code':
            atk = SafeCodeDeadCode(poj, instab, classifier=classifier)
            atk.gen_all(40, opt.iter_select, res_save=save_path + "/data.pkl.gz", origin_data_path=opt.data)
        elif opt.attack_type == 'token':
            atk = SafeCodeToken(poj, symtab, classifier=classifier, device=device)
            atk.gen_all(opt.candidate, opt.iter_select, res_save=save_path + "/data.pkl.gz", origin_data_path=opt.data)
    elif opt.enhance_method == 'random':
        os.makedirs(save_path, exist_ok=True)
        # Random insertion dead code
        if opt.attack_type == 'dead_code':
            atk = RandomDeadCode(poj, instab, classifier=classifier)
            atk.gen_all(40, 20, res_save=save_path + "/data.pkl.gz")
        elif opt.attack_type == 'token':
            atk = RandomToken(poj, symtab, classifier=classifier, device=device)
            atk.gen_all(40, 50, res_save=save_path + "/data.pkl.gz")

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:26:02 2020

@author: DrLC
"""
import autoroot
import argparse
import copy
import gzip
import os
import pickle
import random
import shutil
import time
import sys

import torch
from pycparser import c_parser
import numpy as np
import build_dataset as bd
from dataset import OJ104, CodeChef
from lstm_classifier import LSTMClassifier, LSTMEncoder

from modifier import TokenModifier, InsModifier, token_to_code
from lstm_eval import gettensor, evaluate

import torch
from torch.nn.functional import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mhm_utils import getUID, isUID, getTensor


def similarity(x, y):
    # Cosine similarity calculation
    sim = cosine_similarity(x.unsqueeze(0), y, dim=1)
    # Obtain the coordinates of the maximum
    max_idx = torch.argmax(sim)
    return max_idx


def list_to_ndarray(lst, constant_values=0):
    # Converts a list ndarray
    arr = np.array(lst)
    # If the length of less than 300, the use of zero filling
    if len(arr) < 300:
        arr = np.pad(arr, (0, 300 - len(arr)), 'constant', constant_values=constant_values)
    # If the length of more than 300, only 300 elements
    else:
        arr = arr[:300]
    # Adjust the shape of (1, 300)
    arr = arr.reshape(300, )
    return arr


class Attacker(object):

    def __init__(self, dataset, symtab, classifier, device):

        self.tokenM = TokenModifier(classifier=classifier,
                                    loss=torch.nn.CrossEntropyLoss(),
                                    uids=symtab['all'],
                                    txt2idx=dataset.get_txt2idx(),
                                    idx2txt=dataset.get_idx2txt(),
                                    device=device
                                    )
        self.cl = classifier
        self.d = dataset
        self.syms = symtab
        self.device = device

    def attack(self, x, y, uids, n_candidate=100, n_iter=20):
        iter = 0
        n_stop = 0
        with torch.no_grad():
            old_prob = self.cl.prob(torch.tensor(x, dtype=torch.long).to(self.device).permute([1, 0]))[0]
        if torch.argmax(old_prob) != y[0]:
            # print("SUCC! Original mistake.")
            return True, x, [torch.argmax(old_prob).cpu().numpy()], False
        old_prob = old_prob[y[0]]
        while iter < n_iter:
            keys = list(uids.keys())
            for k in keys:
                if iter >= n_iter:
                    break
                if n_stop >= len(uids):
                    iter = n_iter
                    break
                if k in self.tokenM.forbidden_uid:
                    n_stop += 1
                    continue
                iter += 1
                new_x, new_uid_cand = self.tokenM.rename_uid(x, y, uids[k], k, n_candidate)
                with torch.no_grad():
                    new_prob = self.cl.prob(torch.tensor(new_x, dtype=torch.long).to(self.device).permute([1, 0]))
                new_pred = torch.argmax(new_prob, dim=1)
                for uid, p, pr, _x in zip(new_uid_cand, new_pred, new_prob, new_x):
                    if p != y[0]:
                        return True, [_x], [p.cpu().numpy()], True
                new_prob_idx = torch.argmin(new_prob[:, y[0]])
                if new_prob[new_prob_idx][y[0]] < old_prob:
                    x = [new_x[new_prob_idx]]
                    uids[self.d.idx2vocab(int(new_uid_cand[new_prob_idx]))] = uids.pop(k)
                    n_stop = 0
                    old_prob = new_prob[new_prob_idx][y[0]]
                else:
                    n_stop += 1
        return False, x, y, False

    def attack_all(self, n_candidate=100, n_iter=20, wandb=None):
        n_succ = 0
        total_time = 0
        st_time = time.time()
        asr_success_num = 0
        original_success_num = 0
        for i in range(self.d.test.get_size()):
            b = self.d.test.next_batch(1)
            start_time = time.time()
            tag, adv_x, adv_y, asr_flag = self.attack(b['x'], b['y'], self.syms['te'][b['id'][0]], n_candidate, n_iter)
            if asr_flag:
                asr_success_num += 1
            elif tag and not asr_flag:
                original_success_num += 1
            if tag:
                n_succ += 1
                total_time += time.time() - start_time
            if i % 500 == 0 and i > 0 and wandb is not None:
                # Record against the success rate and average time、ASR
                wandb.log({'token/success_rate': n_succ / (i + 1),
                           'token/asr': asr_success_num / (i - original_success_num),
                           'token/attack_times': i,
                           'token/avg_time_cost': total_time / i,
                           'attack/acc': 1 - (n_succ / (i + 1))
                           }
                          )
                print('asr', asr_success_num / (i - original_success_num))
                print('acc', 1 - (n_succ / (i + 1)))
            elif wandb is None:
                print('wandb is None')

        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (
            time.time() - st_time, n_succ / 3000))


class InsAttacker(object):

    def __init__(self, dataset, instab, classifier, device):

        self.insM = InsModifier(classifier=classifier,
                                txt2idx=dataset.get_txt2idx(),
                                poses=None)  # wait to init when attack
        self.cl = classifier
        self.d = dataset
        self.inss = instab
        self.device = device

    # only support single x: a token-idx list
    def attack(self, x, y, poses, n_candidate=100, n_iter=20):

        self.insM.initInsertDict(poses)

        iter = 0
        n_stop = 0
        old_prob = self.cl.prob(torch.tensor(x, dtype=torch.long).to(self.device).permute([1, 0]))[0]
        if torch.argmax(old_prob) != y[0]:
            return True, x, [torch.argmax(old_prob).cpu().numpy()], False
        old_prob = old_prob[y[0]]
        while iter < n_iter:
            iter += 1

            # get insertion candidates
            n_could_del = self.insM.insertDict["count"]
            n_candidate_del = n_could_del
            n_candidate_ins = n_candidate - n_candidate_del
            assert n_candidate_del >= 0 and n_candidate_ins >= 0
            new_x_del, new_insertDict_del = self.insM.remove(x[0], n_candidate_del)
            new_x_add, new_insertDict_add = self.insM.insert(x[0], n_candidate_ins)
            new_x = new_x_del + new_x_add
            new_insertDict = new_insertDict_del + new_insertDict_add
            if not new_x:  # no valid candidates
                n_stop += 1
                continue

            # find if there is any candidate successful wrong classfied
            feed_new_x = [_x[:self.cl.max_len] for _x in new_x]  # this step is important
            feed_tensor = torch.tensor(feed_new_x, dtype=torch.long)
            new_prob = self.cl.prob(feed_tensor.to(self.device).permute([1, 0]))
            new_pred = torch.argmax(new_prob, dim=1)
            for insD, p, pr, _x in zip(new_insertDict, new_pred, new_prob, new_x):
                if p != y[0]:

                    return True, [_x], [p.cpu().numpy()], True

            # if not, get the one with the lowest target_label_loss
            new_prob_idx = torch.argmin(new_prob[:, y[0]])
            if new_prob[new_prob_idx][y[0]] < old_prob:

                self.insM.insertDict = new_insertDict[new_prob_idx]  # don't forget this step
                n_stop = 0
                old_prob = new_prob[new_prob_idx][y[0]]
            else:
                n_stop += 1
            if n_stop >= len(new_x):  # len(new_x) could be smaller than n_candidate
                iter = n_iter
                break
        return False, x, y, False

    def attack_all(self, n_candidate=100, n_iter=20, wandb=None):
        n_succ = 0
        total_time = 0
        st_time = time.time()
        asr_success_num = 0
        original_pred_error_num = 0
        total_num = self.d.test.get_size()
        for i in range(total_num):
            b = self.d.test.next_batch(1)
            start_time = time.time()
            tag, adv_x, adv_y, asr_success = self.attack(b['x'], b['y'], self.inss['stmt_te'][b['id'][0]], n_candidate,
                                                         n_iter)
            if asr_success:
                asr_success_num += 1
            elif tag and not asr_success:
                original_pred_error_num += 1
            if tag:
                n_succ += 1
                total_time += time.time() - start_time
            # if n_succ <= 0:
            #     print("\tCurr succ rate = %.3f, Avg time cost = NaN sec" \
            #           % (n_succ / (i + 1)), flush=True)
            # else:
            #     print("\tCurr succ rate = %.3f, Avg time cost = %.1f sec" \
            #           % (n_succ / (i + 1), total_time / n_succ), flush=True)
            if i % 200 == 0 and i > 0 and wandb is not None:  # Written to the log
                # Record the current time
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                # Record against the success rate and average time、ASR
                wandb.log({'deadcode/success_rate': n_succ / (i + 1),
                           'deadcode/asr': asr_success_num / (i - original_pred_error_num),
                           'deadcode/attack_times': i,
                           'deadcode/avg_time_cost': total_time / i})
            if i > 2001:
                break

        # if dump_samples_path is not None:
        #     with gzip.open(dump_samples_path, "wb") as f:
        #         pickle.dump(sample_dict, f)
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (
            time.time() - st_time, n_succ / self.d.test.get_size()))


class AttackerRandom(object):

    def __init__(self, dataset, symtab, classifier):

        self.tokenM = TokenModifier(classifier=classifier,
                                    loss=torch.nn.CrossEntropyLoss(),
                                    uids=symtab['all'],
                                    txt2idx=dataset.get_txt2idx(),
                                    idx2txt=dataset.get_idx2txt())
        self.cl = classifier
        self.d = dataset
        self.syms = symtab

    def attack(self, x, y, uids, n_iter=20):

        iter = 0
        n_stop = 0
        old_prob = self.cl.prob(torch.tensor(x, dtype=torch.long).to(device).permute([1, 0]))[0]
        if torch.argmax(old_prob) != y[0]:
            print("SUCC! Original mistake.")
            return True, x, [torch.argmax(old_prob).cpu().numpy()]
        old_prob = old_prob[y[0]]
        while iter < n_iter:
            keys = list(uids.keys())
            for k in keys:
                if iter >= n_iter:
                    break
                if n_stop >= len(uids):
                    iter = n_iter
                    break
                if k in self.tokenM.forbidden_uid:
                    n_stop += 1
                    continue
                iter += 1
                new_x, new_uid_cand = self.tokenM.rename_uid_random(x, uids[k], k)
                new_prob = self.cl.prob(torch.tensor(new_x, dtype=torch.long).to(device).permute([1, 0]))
                new_pred = torch.argmax(new_prob, dim=1)
                for uid, p, pr, _x in zip(new_uid_cand, new_pred, new_prob, new_x):
                    if p != y[0]:
                        print("SUCC!\t%s => %s\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                              (k, self.d.idx2vocab(uid), y[0], old_prob, y[0], pr[y[0]], p, pr[p]))
                        return True, [_x], [p.cpu().numpy()]
                new_prob_idx = torch.argmin(new_prob[:, y[0]])
                if new_prob[new_prob_idx][y[0]] < old_prob:
                    x = [new_x[new_prob_idx]]
                    uids[self.d.idx2vocab(int(new_uid_cand[new_prob_idx]))] = uids.pop(k)
                    n_stop = 0
                    print("acc\t%s => %s\t\t%d(%.5f) => %d(%.5f)" % \
                          (k, self.d.idx2vocab(int(new_uid_cand[new_prob_idx])),
                           y[0], old_prob, y[0], new_prob[new_prob_idx][y[0]]))
                    old_prob = new_prob[new_prob_idx][y[0]]
                else:
                    n_stop += 1
                    print("rej\t%s" % k)
        print("FAIL!")
        return False, x, y

    def attack_all(self, n_iter=20, dump_samples_path=None):

        sample_dict = {"x": [], "y": [], "adv_x": [], "adv_y": []}
        n_succ = 0
        total_time = 0
        st_time = time.time()
        for i in range(self.d.test.get_size()):
            b = self.d.test.next_batch(1)
            sample_dict["x"].append(b['x'][0])
            sample_dict["y"].append(b['y'][0])
            print("\t%d/%d\tID = %d\tY = %d" % (i + 1, self.d.test.get_size(), b['id'][0], b['y'][0]))
            start_time = time.time()
            tag, adv_x, adv_y = self.attack(b['x'], b['y'], self.syms['te'][b['id'][0]], n_iter)
            if tag:
                n_succ += 1
                total_time += time.time() - start_time
                sample_dict["adv_x"].append(adv_x[0])
                sample_dict["adv_y"].append(adv_y[0])
            else:
                sample_dict["adv_x"].append(None)
                sample_dict["adv_y"].append(-1)
            if n_succ <= 0:
                print("\tCurr succ rate = %.3f, Avg time cost = NaN sec" \
                      % (n_succ / (i + 1)), flush=True)
            else:
                print("\tCurr succ rate = %.3f, Avg time cost = %.1f sec" \
                      % (n_succ / (i + 1), total_time / n_succ), flush=True)
        if dump_samples_path is not None:
            with gzip.open(dump_samples_path, "wb") as f:
                pickle.dump(sample_dict, f)
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (
            time.time() - st_time, n_succ / self.d.test.get_size()))


class InsAttackerRandom(object):

    def __init__(self, dataset, instab, classifier):

        self.insM = InsModifier(classifier=classifier,
                                txt2idx=dataset.get_txt2idx(),
                                poses=None)  # wait to init when attack
        self.cl = classifier
        self.d = dataset
        self.inss = instab

    # only support single x: a token-idx list
    def attack(self, x, y, poses, n_iter=20):

        self.insM.initInsertDict(poses)

        iter = 0
        n_stop = 0
        old_prob = self.cl.prob(torch.tensor(x, dtype=torch.long).to(device).permute([1, 0]))[0]
        if torch.argmax(old_prob) != y[0]:
            # print("SUCC! Original mistake.")
            return True, x, [torch.argmax(old_prob).cpu().numpy()]
        old_prob = old_prob[y[0]]
        while iter < n_iter:
            iter += 1

            # get insertion candidates
            new_x, new_insertDict = self.insM.insert_remove_random(x[0])
            if not new_x:  # no valid candidates
                n_stop += 1
                continue

            # find if there is any candidate successful wrong classfied
            feed_new_x = [_x[:self.cl.max_len] for _x in new_x]  # this step is important
            feed_tensor = torch.tensor(feed_new_x, dtype=torch.long)
            new_prob = self.cl.prob(feed_tensor.to(device).permute([1, 0]))
            new_pred = torch.argmax(new_prob, dim=1)
            for insD, p, pr, _x in zip(new_insertDict, new_pred, new_prob, new_x):
                if p != y[0]:
                    # print("SUCC!\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                    #       (self.insM.insertDict["count"], insD["count"],
                    #        y[0], old_prob, y[0], pr[y[0]], p, pr[p]))
                    return True, [_x], [p.cpu().numpy()]

            # if not, get the one with the lowest target_label_loss
            new_prob_idx = torch.argmin(new_prob[:, y[0]])
            if new_prob[new_prob_idx][y[0]] < old_prob:
                # print("acc\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f)" % \
                #       (self.insM.insertDict["count"], new_insertDict[new_prob_idx]["count"],
                #        y[0], old_prob, y[0], new_prob[new_prob_idx][y[0]]))
                self.insM.insertDict = new_insertDict[new_prob_idx]  # don't forget this step
                n_stop = 0
                old_prob = new_prob[new_prob_idx][y[0]]
            else:
                n_stop += 1
                print("rej\t%s" % "")
            if n_stop >= 10:
                iter = n_iter
                break
        print("FAIL!")
        return False, x, y

    def attack_all(self, n_iter=20, dump_samples_path=None):

        sample_dict = {"x": [], "y": [], "adv_x": [], "adv_y": []}
        n_succ = 0
        total_time = 0
        st_time = time.time()
        for i in range(self.d.test.get_size()):
            b = self.d.test.next_batch(1)
            sample_dict["x"].append(b['x'][0])
            sample_dict["y"].append(b['y'][0])
            print("\t%d/%d\tID = %d\tY = %d" % (i + 1, self.d.test.get_size(), b['id'][0], b['y'][0]))
            start_time = time.time()
            tag, adv_x, adv_y = self.attack(b['x'], b['y'], self.inss['stmt_te'][b['id'][0]], n_iter)
            if tag:
                n_succ += 1
                total_time += time.time() - start_time
                sample_dict["adv_x"].append(adv_x[0])
                sample_dict["adv_y"].append(adv_y[0])
            else:
                sample_dict["adv_x"].append(None)
                sample_dict["adv_y"].append(-1)
            if i % 1000 == 0:
                if n_succ <= 0:
                    print("\tCurr succ rate = %.3f, Avg time cost = NaN sec" \
                          % (n_succ / (i + 1)), flush=True)
                else:
                    print("\tCurr succ rate = %.3f, Avg time cost = %.1f sec" \
                          % (n_succ / (i + 1), total_time / n_succ), flush=True)
        if dump_samples_path is not None:
            with gzip.open(dump_samples_path, "wb") as f:
                pickle.dump(sample_dict, f)
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (
            time.time() - st_time, n_succ / self.d.test.get_size()))


class MHMAttacker(object):
    
    def __init__(self, dataset, _classifier, device):
        self.classifier = _classifier
        self.token2idx = dataset.get_txt2idx()
        self.idx2token = dataset.get_idx2txt()
        self.d = dataset
        self.device = device
        
    def attack(self, idx=None, raw_tokens=None, tokens=[], label=None, _n_candi=30,
             _max_iter=100, _prob_threshold=0.95):
        
        if len(raw_tokens) == 0 or len(tokens) == 0 or label is None:
            return None
        raw_seq = ""
        for _t in raw_tokens:
            raw_seq += _t + " "
        tokens_ch = []
        for _t in tokens:
            tokens_ch.append(self.idx2token[_t])
        uid = getUID(tokens_ch)
        if len(uid) <= 0:
            return {'succ': False, 'tokens': None, 'raw_tokens': None}
        for iteration in range(1, 1+_max_iter):
            res = self.__replaceUID(_tokens=tokens, _label=label, _uid=uid,
                                    _n_candi=_n_candi,
                                    _prob_threshold=_prob_threshold)
            # self.__printRes(idx=idx, _iter=iteration, _res=res, _prefix="  >> ")
            if res['status'].lower() in ['s', 'a']:
                tokens = res['tokens']
                uid[res['new_uid']] = uid.pop(res['old_uid'])
                for i in range(len(raw_tokens)):
                    if raw_tokens[i] == res['old_uid']:
                        raw_tokens[i] = res['new_uid']
                if res['status'].lower() == 's':
                    return {'succ': True, 'tokens': tokens,
                            'raw_tokens': raw_tokens}
        return {'succ': False, 'tokens': None, 'raw_tokens': None}

    def attack_all(self, n_candi=30, max_iter=100, prob_threshold=0.95):
        try_cnt = suc_cnt = 0
        p = range(self.d.test.get_size())
        total_time = 0
        bar = tqdm(p, ncols=100, desc='mhm attack')
        acc_cnt = 0
        for i in bar:
            _b = self.d.test.next_batch(1)
            start_time = time.time()
            inputs, labels = getTensor(_b)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                outputs = self.classifier(inputs)[0]
                origin_mistake = torch.argmax(outputs, dim=1) != labels
            if origin_mistake:
                continue
            try_cnt += 1
            _res = self.attack(idx=i, raw_tokens=_b['raw'][0], tokens=_b['x'][0],
                             label=_b['y'][0], _n_candi=n_candi,
                             _max_iter=max_iter, _prob_threshold=prob_threshold)
            total_time += time.time() - start_time
            if _res['succ']:
                suc_cnt += 1
            else:
                acc_cnt += 1
            bar.set_description('asr = ' + str(round(suc_cnt / try_cnt * 100, 2)) + '(' + \
                        str(suc_cnt) + '/' + str(try_cnt) + ')' + \
                        ', acc = ' + str(round(acc_cnt / (i+1) * 100, 2)) + \
                        '(' + str(acc_cnt) + '/' + str(i+1) + ')')
        
    def __replaceUID(self, _tokens=[], _label=None, _uid={},
                     _n_candi=30, _prob_threshold=0.95, _candi_mode="random"):
        
        assert _candi_mode.lower() in ["random", "nearby"]
        
        selected_uid = random.sample(_uid.keys(), 1)[0]
        if _candi_mode == "random":
            # First, generate candidate set.
            # The transition probabilities of all candidate are the same.
            candi_token = [selected_uid]
            candi_tokens = [copy.deepcopy(_tokens)]
            candi_labels = [_label]
            for c in random.sample(self.idx2token, _n_candi):
                if isUID(c):
                    candi_token.append(c)
                    candi_tokens.append(copy.deepcopy(_tokens))
                    candi_labels.append(_label)
                    for i in _uid[selected_uid]:
                        if i >= len(candi_tokens[-1]):
                            break
                        candi_tokens[-1][i] = self.token2idx[c]
            # Then, feed all candidates into the model
            _candi_tokens = numpy.asarray(candi_tokens)
            _candi_labels = numpy.asarray(candi_labels)
            _inputs, _labels = getTensor({"x": _candi_tokens,
                                          "y": _candi_labels}, False)
            _inputs = _inputs.to(self.device)
            _labels = _labels.to(self.device)
            prob = self.classifier.prob(_inputs)
            pred = torch.argmax(prob, dim=1)
            for i in range(len(candi_token)):   # Find a valid example
                if pred[i] != _label:
                    return {"status": "s", "alpha": 1, "tokens": candi_tokens[i],
                            "old_uid": selected_uid, "new_uid": candi_token[i],
                            "old_prob": prob[0], "new_prob": prob[i],
                            "old_pred": pred[0], "new_pred": pred[i]}
            candi_idx = torch.argmin(prob[1:, _label]) + 1
            candi_idx = int(candi_idx.item())
            # At last, compute acceptance rate.
            prob = prob.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            alpha = (1-prob[candi_idx][_label]+1e-10) / (1-prob[0][_label]+1e-10)
            if random.uniform(0, 1) > alpha or alpha < _prob_threshold:
                return {"status": "r", "alpha": alpha, "tokens": candi_tokens[i],
                        "old_uid": selected_uid, "new_uid": candi_token[i],
                        "old_prob": prob[0], "new_prob": prob[i],
                        "old_pred": pred[0], "new_pred": pred[i]}
            else:
                return {"status": "a", "alpha": alpha, "tokens": candi_tokens[i],
                        "old_uid": selected_uid, "new_uid": candi_token[i],
                        "old_prob": prob[0], "new_prob": prob[i],
                        "old_pred": pred[0], "new_pred": pred[i]}
        else:
            pass

    def __printRes(self, idx=None, _iter=None, _res=None, _prefix="  => "):
        if _res['status'].lower() == 's':   # Accepted & successful
            print("%s iter %d, SUCC! %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                   (_prefix, _iter, _res['old_uid'], _res['new_uid'], \
                    _res['old_pred'], _res['new_pred'], \
                    _res['old_prob'][_res['old_pred']], \
                    _res['new_prob'][_res['old_pred']], _res['alpha']))
        elif _res['status'].lower() == 'r': # Rejected
            print("%s iter %d, REJ. %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                   (_prefix, _iter, _res['old_uid'], _res['new_uid'], \
                    _res['old_pred'], _res['new_pred'], \
                    _res['old_prob'][_res['old_pred']], \
                    _res['new_prob'][_res['old_pred']], _res['alpha']))
        elif _res['status'].lower() == 'a': # Accepted
            print("%s iter %d, ACC! %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                   (_prefix, _iter, _res['old_uid'], _res['new_uid'],
                    _res['old_pred'], _res['new_pred'],
                    _res['old_prob'][_res['old_pred']],
                    _res['new_prob'][_res['old_pred']], _res['alpha']))

if __name__ == "__main__":
    device = torch.device("cuda:" + opt.gpu)
    root = '../data/'
    embedding_size = 512
    hidden_size = 600
    n_layers = 2
    model_name = "best.pt"
    model = "LSTM"
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--attn', action='store_true')
    parser.add_argument('--task', type=str, default='code_function', help='code_function, code_defect')
    parser.add_argument('--model', default=model)
    parser.add_argument('--model_parameter', default='')
    parser.add_argument('--attack_type', default='token', help='token, dead_code')
    parser.add_argument('--model_save_path',
                        default="")
    parser.add_argument('--server', default='2229', type=str, required=False)
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    if opt.task == 'code_function':
        n_class = 104
        max_len = 500
        vocab_size = 5000
        opt.data = root + opt.task + '/dataset/origin/data.pkl.gz'
        original_dataset = OJ104(path=opt.data)
        training_s = original_dataset.train
        valid_s = original_dataset.dev
        test_s = original_dataset.test
    elif opt.task == 'code_defect':
        n_class = 4
        max_len = 300
        vocab_size = 3000
        opt.data = root + opt.task + '/dataset/origin/data.pkl.gz'
        original_dataset = CodeChef(path=opt.data)
        training_s = original_dataset.train
        valid_s = original_dataset.dev
        test_s = original_dataset.test

    model_parameter = opt.model_parameter
    model = opt.model

    model_save_path = opt.model_save_path
    with gzip.open(os.path.join(root, opt.task, 'data/origin/data_uid.pkl.gz'), "rb") as f:
        symtab = pickle.load(f)
    with gzip.open(os.path.join(root, opt.task, 'data/origin/data_inspos.pkl.gz'), "rb") as f:
        instab = pickle.load(f)

    enc = LSTMEncoder(embedding_size, hidden_size, n_layers)
    classifier = LSTMClassifier(vocab_size, embedding_size, enc,
                                hidden_size, n_class, max_len, attn=opt.attn).to(device)
    if opt.model_save_path is None:
        model_save_path = "/data_raw/yjy/code/2023/paper01_data/code_defect/model/LSTM/ALERT/origin+alert_token/best.pt"
    else:
        model_save_path = opt.model_save_path
    print(model_save_path)
    test = torch.load(model_save_path)
    classifier.load_state_dict(torch.load(model_save_path))
    log_path = os.path.dirname(model_save_path)
    # evaluate(classifier, test_s, log_path, batch_size=128)
    classifier.train()

    if opt.attack_type == 'token':
        print('token attack:' + model_save_path)
        atk = Attacker(original_dataset, symtab, classifier)
        atk.attack_all(40, 50)
    elif opt.attack_type == 'dead_code':
        atk = InsAttacker(original_dataset, instab, classifier)
        atk.attack_all(40, 20)
    elif opt.attack_type == 'all':
        atk = Attacker(original_dataset, symtab, classifier)
        atk.attack_all(40, 50)
        atk = InsAttacker(original_dataset, instab, classifier)
        atk.attack_all(40, 20)
    elif opt.attack_type == 'mhm':
        atk = MHMAttacker(original_dataset, classifier, device)
        atk.attack_all(40, 400, 1)

    # =========================== attack after adv-training =========================

    # atk = TDAttacker(original_dataset, symtab, instab, classifier)
    # atk.attack_all(40, 50, 20)

    # atk = AttackerRandom(original_dataset, symtab, classifier)
    # atk.attack_all(100)

    # atk = InsAttackerRandom(original_dataset, instab, classifier)
    # atk.attack_all(20)

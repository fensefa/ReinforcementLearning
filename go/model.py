#!usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import random
import time
import os
from functools import reduce


class BaseModel():
    def __init__(self, go_size):
        self._go_size = go_size
    def predict(self, cur_state):
        raise NotImplementedError
    def result(self, cur_state):
        raise NotImplementedError

class FiveModel(BaseModel):
    def result(self, cur_state, i=-1,j=-1):
        # t0 = time.time()
        if np.all(cur_state != 0):
            return True, 0, -1, -1
        # t1 = time.time()
        if i == -1 or j == -1:
            return self.result1(cur_state)
        else:
            return self.result0(cur_state,i,j)
        return r0
        t2 = time.time()
        r1 = self.result1(cur_state)
        t3 = time.time()
        if r0 != r1:
            print(cur_state)
            print("r0", r0)
            print("r1", r1)
            exit()
        print(r0 == r1, (t1-t0)*1000,(t2-t1)*1000,(t3-t2)*1000)
        return r1
        return r0
    def result0(self, cur_state, idx_i, idx_j):
        for i in range(max(0, idx_i - 4), min(idx_i+1, self._go_size - 4)):
            temp = np.sum(cur_state[i:i+5,idx_j])
            if temp == 5:
                return True, 1, i, idx_j
            elif temp == -5:
                return False, 1, i, idx_j
        for j in range(max(0, idx_j - 4), min(idx_j+1, self._go_size - 4)):
            temp = np.sum(cur_state[idx_i,j:j+5])
            if temp == 5:
                return True, 2, idx_i, j
            elif temp == -5:
                return False, 2, idx_i, j
        for k in range(max(-4, -idx_i, -idx_j), min(1, self._go_size - idx_i - 4, self._go_size - idx_j - 4)):
            temp = sum([cur_state[idx_i+k+l,idx_j+k+l] for l in range(5)])
            if temp == 5:
                return True, 3, idx_i+k, idx_j+k
            elif temp == -5:
                return False, 3, idx_i+k, idx_j+k
        for k in range(max(-4, -idx_i, idx_j-self._go_size+1), min(1, self._go_size-idx_i-4, idx_j-3)):
            temp = sum([cur_state[idx_i+k+l,idx_j-k-l] for l in range(5)])
            if temp == 5:
                return True, 4, idx_i+k, idx_j-k
            elif temp == -5:
                return False, 4, idx_i+k, idx_j-k
        return None, None, -1, -1
    def result1(self, cur_state):
        for i in range(self._go_size - 4):
            for j in range(self._go_size):
                temp = np.sum(cur_state[i:i+5,j])
                if temp == 5:
                    return True, 1, i, j
                elif temp == -5:
                    return False, 1, i, j
        for i in range(self._go_size):
            for j in range(self._go_size - 4):
                temp = np.sum(cur_state[i,j:j+5])
                if temp == 5:
                    return True, 2, i, j
                elif temp == -5:
                    return False, 2, i, j
        for i in range(self._go_size - 4):
            for j in range(self._go_size - 4):
                temp = sum([cur_state[i+k,j+k] for k in range(5)])
                if temp == 5:
                    return True, 3, i, j
                elif temp == -5:
                    return False, 3, i, j
        for i in range(self._go_size - 4):
            for j in range(4, self._go_size):
                temp = sum([cur_state[i+k,j-k] for k in range(5)])
                if temp == 5:
                    return True, 4, i, j
                elif temp == -5:
                    return False, 4, i, j
        return None, None, -1, -1

class RuleModel(FiveModel):
    def __init__(self, go_size):
        super().__init__(go_size)
    def predict(self, cur_state):
        for i in range(self._go_size):
            for j in range(self._go_size):
                if cur_state[i][j] == 0:
                    return i,j
        return 0,0

class DqnBaseModel(FiveModel):
    def __init__(self, go_size):
        super().__init__(go_size)
        self._state_ph = tf.placeholder(tf.float32, name="state_ph", shape=[None, self._go_size, self._go_size])
        self._index_ph = tf.placeholder(tf.int32, name="index_ph", shape=[None])
        self._score_ph = tf.placeholder(tf.float32, name="score_ph", shape=[None])
        self._drop_ph = tf.placeholder(tf.float32, name="drop_ph", shape=[])
    def build_model(self):
        scores = self.make_logit(self._state_ph)
        print("state_ph", self._state_ph.shape)
        print("scores", scores.shape)
        print("score_ph", self._score_ph.shape)
        index_one_hot = tf.one_hot(self._index_ph, self._go_size * self._go_size)
        print("index_one_hot", index_one_hot.shape)
        action_score = tf.einsum("ij,ij->i", scores, index_one_hot)
        print("action_score", action_score.shape)
        loss = tf.square(action_score - self._score_ph)
        self._loss = tf.reduce_mean(loss)
        self._train_op = tf.train.AdamOptimizer(5e-4).minimize(self._loss)
        
        predict_mask = tf.cast(tf.cast(self._state_ph, tf.bool), tf.float32)
        predict_mask = tf.reshape(predict_mask, tf.shape(scores))
        self._scores = scores - predict_mask * 9
        print("scores", self._scores.shape)
        
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        self.print_variables()
        self._saver = tf.train.Saver(tf.global_variables())
        self._global_step = -1
        self._cur_data_size = 0
    def save(self, path="model/dnn_model"):
        self._saver.save(self._sess, path)
    def restore(self, path="model/dnn_model"):
        self._saver.restore(self._sess, path)
    def make_logit(self, state):
        raise NotImplementedError
    def train(self):
        batch_size = 128
        train_batch = 1000000
        train_data_size = 1000000
        train_state_data = np.zeros([train_data_size, self._go_size, self._go_size])
        train_index_data = np.zeros([train_data_size], dtype=np.int32)
        train_score_data = np.zeros([train_data_size])
        max_data_size = 100000
        for global_step in tqdm(range(train_batch), ascii=True, desc="Train "):
            self._global_step = global_step
            t0 = time.time() * 1000
            dataset = self.play()
            t1 = time.time() * 1000
            # print("Play_cost:", t1 - t0)
            if self._cur_data_size < train_data_size:
                self._cur_data_size = min(self._cur_data_size + len(dataset), train_data_size)
                idx = range(self._cur_data_size - len(dataset), self._cur_data_size)
            else:
                idx = [random.randint(0, train_data_size-1) for _ in range(len(dataset))]
            train_state_data[idx] = [_[0] for _ in dataset]
            train_index_data[idx] = [_[1] for _ in dataset]
            train_score_data[idx] = [_[2] for _ in dataset]
            t2 = time.time() * 1000
            # print("Data_cost:", t2 - t1)
            for j in range(1):
                idx = [random.randint(0, self._cur_data_size-1) for _ in range(batch_size)]
                state = train_state_data[idx]
                index = train_index_data[idx]
                score = train_score_data[idx]
                loss,_ = self._sess.run([self._loss, self._train_op], feed_dict = {
                    self._state_ph: state,
                    self._index_ph: index,
                    self._score_ph: score,
                    self._drop_ph: .3,
                })
            t3 = time.time() * 1000
            # print("Train_cost:", t3 - t2)
            if (global_step+1) % 100 == 0:
                state = train_state_data[:1000]
                index = train_index_data[:1000]
                score = train_score_data[:1000]
                loss = self._sess.run(self._loss, feed_dict = {
                    self._state_ph: state,
                    self._index_ph: index,
                    self._score_ph: score,
                    self._drop_ph: .0,
                    })
                print("Evaluate:", .9995 ** self._global_step, np.count_nonzero(train_score_data), loss)
                self.save()
            if (global_step+1) % 100 == 0:
                with open("data/play_%06d" % (global_step+1), "w") as fh:
                    for state, index, score in dataset:
                        scores = self._sess.run(self._scores, feed_dict={
                            self._state_ph: [state],
                            self._drop_ph: .0,
                            })
                        fh.write('='*30 + '\n')
                        fh.write('+' + ''.join(["%3d" % _ for _ in range(self._go_size)]) + '\n')
                        k = 0
                        for line in state:
                            fh.write("%1d" % k + ''.join(["%3d" % _ for _ in line]) + '\n')
                            k+=1
                        fh.write('\n{}\t{}\t{}\n'.format(index//self._go_size, index%self._go_size, score))
                        fh.write('\n+' + ''.join(["%6d" % _ for _ in range(self._go_size)]) + '\n')
                        for i in range(self._go_size):
                            l = i * self._go_size
                            fh.write("%1d  " % i + ''.join(["% 6.2f" % _ for _ in scores[0][l:l+self._go_size].tolist()]) + '\n')
                        index = np.argmax(scores)
                        fh.write('\n{}\t{}\t{}\n'.format(index//self._go_size, index%self._go_size, scores[0][index]))
    def play(self):
        t0 = time.time() * 1000
        if self._global_step % 100 == 0:
            print("*"*50)
        state = np.zeros([self._go_size, self._go_size])
        t1 = time.time() * 1000
        black_data, white_data = [], []
        res = None
        while res == None:
            res = self.step(state, black_data, white_data)
        t2 = time.time() * 1000
        dataset = []
        for k in range(len(white_data) - 1):
            dataset.append((black_data[k][0], black_data[k][1], black_data[k+1][2] * .95))
            dataset.append((white_data[k][0], white_data[k][1], white_data[k+1][2] * .95))
        if len(white_data) < len(black_data):
            dataset.append((black_data[-2][0], black_data[-2][1], black_data[-1][2] * .95))
            dataset.append((white_data[-1][0], white_data[-1][1], -1))
            dataset.append((black_data[-1][0], black_data[-1][1], 1))
        else:
            dataset.append((black_data[-1][0], black_data[-1][1], -1))
            dataset.append((white_data[-1][0], white_data[-1][1], 1))
        t3 = time.time() * 1000
        # print("Plays:", t1-t0,t2-t1,t3-t2)
        return dataset
    def step(self, state, black_data, white_data):
        # 黑棋
        i, j, score = self.epison_greedy(state)
        black_data.append((state.copy(), i*self._go_size+j, score))
        assert state[i][j] == 0
        state[i][j] = 1
        res1 = self.result(state, i, j)
        if res1[0] != None:
            return res1[0]
        # 白棋
        t0 = time.time() * 1000
        i, j, score = self.epison_greedy(-state)
        t1 = time.time() * 1000
        white_data.append((-state.copy(), i*self._go_size+j, score))
        t2 = time.time() * 1000
        assert state[i][j] == 0
        state[i][j] = -1
        t3 = time.time() * 1000
        res2 = self.result(state, i, j)
        t4 = time.time() * 1000
        # print("Steps:", t1-t0,t2-t1,t3-t2,t4-t3)
        return res2[0]
    def epison_greedy(self, state):
        res = self.predict(state)
        eps = max(.1, .9997 ** self._global_step)
        if random.random() < eps:
            idx = random.randint(0, self._go_size * self._go_size - 1)
            i = idx // self._go_size
            j = idx % self._go_size
            if state[i][j] == 0:
                return i, j, res[2]
        return res
    def predict(self, cur_state):
        scores = self._sess.run(self._scores, feed_dict={
            self._state_ph: [cur_state],
            self._drop_ph: .0,
            })
        idx = np.argmax(scores)
        score = np.max(scores)
        return idx // self._go_size, idx % self._go_size, score
    def print_variables(self):
        total_volumn = 0
        for v in tf.trainable_variables():
            volumn = reduce(lambda x,y:x*y, v.get_shape().as_list(), 1)
            total_volumn += volumn
            print(v.name, v.shape, volumn)
        print("Total", total_volumn)
class DNNModel(DqnBaseModel):
    def __init__(self, go_size):
        l = 2
        self._func_list = []
        for size in [50, 20, 10, 5]:
            dense = tf.keras.layers.Dense(size, activation=tf.nn.relu, use_bias=True)
            self._func_list.append(dense)
        dense = tf.keras.layers.Dense(2, activation=None, use_bias=False)
        self._func_list.append(dense)
        super().__init__(go_size)
    def make_logit(self, state):
        hidden = tf.reshape(state, [tf.shape(state)[0], self._go_size * self._go_size])
        for func in self._func_list:
            hidden = func(hidden)
        return logits
class CNNModel(DqnBaseModel):
    def __init__(self, go_size):
        super().__init__(go_size)
        self._func_list = []
        for size in [5,3,3,2]:
            cnn = tf.keras.layers.Conv2D(128, (size, size), activation=tf.nn.relu, use_bias=True)
            self._func_list.append(cnn)
            drop = tf.keras.layers.Dropout(self._drop_ph)
            self._func_list.append(drop)
        dense = tf.keras.layers.Dense(self._go_size * self._go_size, activation=None, use_bias=False)
        self._func_list.append(dense)
    def make_logit(self, state):
        hidden = tf.reshape(state, [tf.shape(state)[0], self._go_size, self._go_size, 1])
        for func in self._func_list:
            print("hidden", hidden.shape)
            hidden = func(hidden)
        print("hidden", hidden.shape)
        logits = tf.squeeze(hidden, axis=[1,2])
        print("logits", logits.shape)
        return logits

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = CNNModel(10)
    model.build_model()
    model.train()

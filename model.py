#!usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import random
import time
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
            return False, 0, -1, -1
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
        self._score_ph = tf.placeholder(tf.float32, name="score_ph", shape=[None])
        with tf.variable_scope("logits"):
            logits = self.make_logit(self._state_ph)
        print("state_ph", self._state_ph.shape)
        print("logits", logits.shape)
        print("score_ph", self._score_ph.shape)
        loss = tf.square(logits - self._score_ph)
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self._score_ph, logits=logits)
        self._loss = tf.reduce_mean(loss)
        self._train_op = tf.train.AdamOptimizer(1e-4).minimize(self._loss)
        predict_inputs = tf.tile(self._state_ph, [self._go_size * self._go_size, 1, 1])
        print("predict_inputs", predict_inputs.shape)
        position_inputs = tf.eye(self._go_size * self._go_size)
        position_inputs = tf.reshape(position_inputs, [self._go_size * self._go_size, self._go_size, self._go_size])
        final_inputs = predict_inputs + position_inputs
        print("final_inputs", final_inputs.shape)
        # final_inputs = tf.Print(final_inputs, [predict_inputs[0]], summarize=10000, message="Predict_inputs:\n")
        # final_inputs = tf.Print(final_inputs, [final_inputs[0]], summarize=10000, message="Final_inputs0:\n")
        # final_inputs = tf.Print(final_inputs, [final_inputs[1]], summarize=10000, message="Final_inputs1:\n")
        # final_inputs = tf.Print(final_inputs, [final_inputs], 'final_inputs')
        with tf.variable_scope("logits"):
            scores = self.make_logit(final_inputs)
        predict_mask = tf.cast(tf.cast(self._state_ph, tf.bool), tf.float32)
        predict_mask = tf.reshape(predict_mask, tf.shape(scores))
        # scores = tf.Print(scores, [scores], 'scores')
        self._scores = scores - predict_mask * 10
        # self._scores = tf.Print(self._scores, [self._scores], summarize=10000, message="Scores:\n")
        print("scores", self._scores.shape)
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        self.print_variables()
        self._saver = tf.train.Saver(tf.global_variables())
        self._global_step = -1
    def save(self, path="model/dnn_model"):
        self._saver.save(self._sess, path)
    def restore(self, path="model/dnn_model"):
        self._saver.restore(self._sess, path)
    def make_logit(self, state):
        raise NotImplementedError
    def train(self):
        batch_size = 128
        train_batch = 1000000
        train_data_size = 100000
        train_state_data = np.zeros([train_data_size, self._go_size, self._go_size])
        train_score_data = np.zeros([train_data_size])
        max_data_size = 100000
        for global_step in tqdm(range(train_batch), ascii=True, desc="Train "):
            self._global_step = global_step
            t0 = time.time() * 1000
            dataset = self.play()
            t1 = time.time() * 1000
            # print("Play_cost:", t1 - t0)
            idx = [random.randint(0, train_data_size-1) for _ in range(len(dataset))]
            train_state_data[idx] = [_[0] for _ in dataset]
            train_score_data[idx] = [_[3] for _ in dataset]
            t2 = time.time() * 1000
            # print("Data_cost:", t2 - t1)
            for j in range(1):
                idx = [random.randint(0, train_data_size-1) for _ in range(batch_size)]
                state = train_state_data[idx]
                score = train_score_data[idx]
                loss,_ = self._sess.run([self._loss, self._train_op], feed_dict = {
                    self._state_ph: state,
                    self._score_ph: score,
                })
            t3 = time.time() * 1000
            # print("Train_cost:", t3 - t2)
            if (global_step+1) % 100 == 0:
                state = train_state_data[:1000]
                score = train_score_data[:1000]
                loss = self._sess.run(self._loss, feed_dict = {
                    self._state_ph: state,
                    self._score_ph: score,
                    })
                print("Evaluate:", .9995 ** self._global_step, np.count_nonzero(train_score_data), loss)
                self.save()
            if (global_step+1) % 100 == 0:
                with open("data/play_%06d" % (global_step+1), "w") as fh:
                    for state, i, j, res in dataset:
                        fh.write('='*30 + '\n')
                        fh.write('+' + ''.join(["%3d" % _ for _ in range(self._go_size)]) + '\n')
                        k = 0
                        for line in state:
                            fh.write("%1d" % k + ''.join(["%3d" % _ for _ in line]) + '\n')
                            k+=1
                        fh.write('\n{}\t{}\t{}\n'.format(i,j,res))
    def play(self):
        t0 = time.time() * 1000
        if self._global_step % 100 == 0:
            print("*"*50)
        state = np.zeros([self._go_size, self._go_size])
        t1 = time.time() * 1000
        res = None
        black_data, white_data = [], []
        dataset = []
        while res == None:
            res = self.step(state, black_data, white_data)
        t2 = time.time() * 1000
        score = 1 if res else -1
        for k in range(len(black_data)):
            dataset.append((*black_data[k], score * .95 ** (len(black_data)-k-1)))
        for k in range(len(white_data)):
            dataset.append((*white_data[k], -score * .95 ** (len(white_data)-k-1)))
        t3 = time.time() * 1000
        # print("Plays:", t1-t0,t2-t1,t3-t2)
        return dataset
    def step(self, state, black_data, white_data):
        # 黑旗
        i, j = self.epison_greedy(state)
        black_data.append((state.copy(), i,j))
        assert state[i][j] == 0
        state[i][j] = 1
        res1 = self.result(state, i, j)
        if res1[0] != None:
            return res1[0]
        # 白旗
        t0 = time.time() * 1000
        i, j = self.epison_greedy(-state)
        t1 = time.time() * 1000
        white_data.append((-state.copy(), i,j))
        t2 = time.time() * 1000
        assert state[i][j] == 0
        state[i][j] = -1
        t3 = time.time() * 1000
        res2 = self.result(state, i, j)
        t4 = time.time() * 1000
        # print("Steps:", t1-t0,t2-t1,t3-t2,t4-t3)
        return res2[0]
    def epison_greedy(self, state):
        res1 = self.predict(state)
        eps = max(.1, .9997 ** self._global_step)
        if random.random() < eps:
            idx = random.randint(0, self._go_size * self._go_size - 1)
            i = idx // self._go_size
            j = idx % self._go_size
            if state[i][j] == 0:
                return i, j
        return res1
    def predict(self, cur_state):
        scores = self._sess.run(self._scores, feed_dict={self._state_ph: [cur_state]})
        idx = np.argmax(scores)
        if self._global_step % 100 == 0:
            print('='*20)
            print("State:\n", cur_state)
            print("Index:", idx // self._go_size, idx % self._go_size)
            print("Score:\n", np.reshape(scores, [self._go_size, self._go_size]))
        return idx // self._go_size, idx % self._go_size
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
        dense = tf.keras.layers.Dense(1, activation=None, use_bias=False)
        self._func_list.append(dense)
        super().__init__(go_size)
    def make_logit(self, state):
        l = 2
        hidden1 = tf.reshape(state, [tf.shape(state)[0], self._go_size * self._go_size])
        hidden2 = -hidden1
        hidden = tf.concat((hidden1, hidden2), axis=0)
        for func in self._func_list:
            hidden = func(hidden)
        # logits = tf.nn.sigmoid(hidden)
        logits = tf.squeeze(hidden, axis=1)
        logits1, logits2 = tf.split(logits, 2)
        return (logits1 - logits2 + 1) / 2
        return logits1
        for size in [50, 20, 10, 5]:
            hidden = tf.layers.dense(hidden, size, activation=tf.nn.relu, use_bias=True)
        logits = tf.layers.dense(hidden, 1, activation=None)
        # logits = tf.nn.sigmoid(logits)
        logits = tf.squeeze(logits, axis=1)
        logits1, logits2 = tf.split(logits, 2)
        return logits1
        return (logits1 - logits2 + 1) / 2
class CNNModel(DqnBaseModel):
    def __init__(self, go_size):
        l = 2
        self._func_list = []
        for size in [5,3,3,2]:
            cnn = tf.keras.layers.Conv2D(64, (size, size), activation=tf.nn.relu, use_bias=True)
            self._func_list.append(cnn)
        '''
        pool = tf.keras.layers.MaxPool2D(2,2)
        self._func_list.append(pool)
        '''
        dense = tf.keras.layers.Dense(1, activation=None, use_bias=False)
        self._func_list.append(dense)
        super().__init__(go_size)
    def make_logit(self, state):
        l = 2
        hidden1 = tf.reshape(state, [tf.shape(state)[0], self._go_size, self._go_size, 1])
        hidden2 = -hidden1
        hidden = tf.concat((hidden1, hidden2), axis=0)
        for func in self._func_list:
            print("hidden", hidden.shape)
            hidden = func(hidden)
        print("hidden", hidden.shape)
        logits = tf.squeeze(hidden, axis=[1,2,3])
        print("logits", logits.shape)
        logits1, logits2 = tf.split(logits, 2)
        return (logits1 - logits2) / 2

if __name__ == "__main__":
    model = CNNModel(10)
    model.train()

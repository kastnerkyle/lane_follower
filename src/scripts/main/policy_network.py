#!/usr/bin/env python

""" Trains an agent with (stochastic) Policy Gradients on line follower"""
import numpy as np
import cPickle as pickle
import gym

class DPN(object):
    """
    Deep Ploci Network Reinforcement Learning Class
    """

    def __init__(self, lr, y, batch_size, decay_rate):
        # hyper-parameters
        self.lr = lr
        self.y = y
        self.batch_size = batch_size
        self.decay_rate = decay_rate

        # simply two-layer model
        self.model = {}
        model['W1'] = np.random.randn(200, 32*32) / np.sqrt(32*32)
        model['W2'] = np.random.randn(200) / np.sqrt(200)

        grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
        rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory


    def sigmoid(x): 
        return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

    def policy_forward(x):
        h = np.dot(model['W1'], x)
        h[h<0] = 0 # ReLU nonlinearity
        logp = np.dot(model['W2'], h)
        p = sigmoid(logp)
        return p, h # return probability of taking action 2, and hidden state

    def policy_backward(eph, epdlogp):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, model['W2'])
        dh[eph <= 0] = 0 # backpro prelu
        dW1 = np.dot(dh.T, epx)
        return {'W1':dW1, 'W2':dW2}

    def start(self):
        pass

    def stop(self):
        pass

    def update(self, state):
        pass

    def feed_forward(self, state):
        pass

    def calculate_reward(self, binary):
        pass
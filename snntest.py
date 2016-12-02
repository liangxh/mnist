#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xihao liang
@created: 2016.11.29
"""

import loader

import math
import h5py
import numpy as np
from optparse import OptionParser
from keras.models import model_from_json
import progbar

from graph import Canvas


class IFLayer:
    def __init__(self, W):
        self.W = W
        self.input_dim = self.W.shape[0]
        self.output_dim = self.W.shape[1]

    def reset(self, n_sample):
        self.mem = np.zeros((n_sample, self.output_dim))

    def sim(self, input_spikes, thr):
        dmem = np.dot(input_spikes, self.W)

        self.mem += dmem
        spikes_idx = self.mem >= thr

        # reset
        self.mem[spikes_idx] = 0.

        # build output
        output_spikes = np.zeros(self.mem.shape)
        output_spikes[spikes_idx] = 1.
        
        return output_spikes

    @staticmethod
    def from_relu(W):
        '''max_input_sum = 0.

        for i in range(W.shape[1]):
            w = W[:, i]
            input_sum = np.sum(w[w > 0.])           
            max_input_sum = max(max_input_sum, input_sum)

        W /= max_input_sum'''
        return IFLayer(W)


class SNN:
    def __init__(self):
        self.layers = []
        self.thr = 1.0
        self.dt = 0.001
        self.duration = 0.040
        self.max_rate = 200.
        self.input_dim = 784
        self.o = 0.

    def add(self, layer):
        self.layers.append(layer)

    def reset(self, input_dim):
        for layer in self.layers:
            layer.reset(input_dim)

    def classify(self, X, batch_size = 128, verbose=1):
        idx_start = 0
        n_sample = X.shape[0]
        pred = []
        
        if verbose == 1:
            pbar = progbar.start(int(math.ceil(float(n_sample) / batch_size)))
            i = 0
        
        while idx_start < n_sample:
            idx_end = idx_start + batch_size
            pred.extend(self.classify_batch(X[idx_start:idx_end, :]).tolist())

            idx_start += batch_size

            if verbose == 1:
                i += 1
                pbar.update(i)

        if verbose == 1:
            pbar.finish()

        return pred
        
    def classify_batch(self, X):
        n_sample = X.shape[0]
        self.reset(n_sample)
        
        n_timestep = int(math.ceil(self.duration / self.dt))
        rescale_fac = 1. / (self.dt * self.max_rate);

        spike_sum = np.zeros((n_sample, 10))

        #canvas = Canvas(5, 8)
        for t in range(n_timestep):
            spike_snapshot = np.random.random(X.shape) * rescale_fac
            
            spikes = np.zeros(X.shape)
            spikes[spike_snapshot <= X] = 1.
            #canvas.draw(spikes)

            for i, layer in enumerate(self.layers):
                spikes = layer.sim(spikes, self.thr)

            spike_sum += spikes

        #canvas.show()
        c = np.sum(spike_sum ** 2, axis = 1)
        self.o += np.sum(c == 0)

        return np.argmax(spike_sum, axis = 1)


def main():
    optparser = OptionParser()
    optparser.add_option('-n', '--n', dest='n', type='int', default=None)
    optparser.add_option('-b', '--bias', dest='bias', type='int', default=1)
    opts, args = optparser.parse_args()

    model_name = 'snn'
    fname_weight = 'model/%s_%d_weights.hdf5'%(model_name, opts.bias)
    fname_config = 'model/%s_%d_config.json'%(model_name, opts.bias)

    f = h5py.File(fname_weight, 'r')

    snn = SNN()
    for i in range(1, 4):
        layer_name = u'dense_%d' % i
        W = f[layer_name][layer_name + '_W']
        snn.add(IFLayer.from_relu(W))

    test = loader.load_test(opts.n)
    imgs, gold = test

    func_adapt = __import__(model_name, fromlist=['']).__dict__['adapt']
    X, Y = func_adapt(test)

    pred = snn.classify(X)
    print float(np.sum(pred == gold)) / len(gold)

if __name__ == '__main__':
    main()

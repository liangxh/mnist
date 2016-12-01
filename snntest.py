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

def adapt(XY):
    X, Y = XY
    X = np.asarray(map(lambda x:x.flatten(), X))
    Y = np_utils.to_categorical(map(int, Y), 10)

    return X, Y


class IFLayer:
    def __init__(self, W):
        self.W = W
        self.input_dim = self.W.shape[0]
        self.output_dim = self.W.shape[1]
        self.reset()

    def reset(self):
        self.mem = np.zeros(self.output_dim)

    def sim(self, input_spikes, thr):
        dmem = np.dot(input_spikes, self.W)
        #print dmem

        self.mem += dmem
        #print self.mem[:15]
        spikes_idx = self.mem >= thr

        # reset
        self.mem[spikes_idx] = 0.

        # build output
        output_spikes = np.zeros(self.output_dim)
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
        self.duration = 0.04
        self.max_rate = 400.
        self.input_dim = 784

    def add(self, layer):
        self.layers.append(layer)

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def classify(self, x):
        self.reset()

        x = np.asarray(x) / 255.
        
        n_timestep = int(math.ceil(self.duration / self.dt))
        rescale_fac = 1. / (self.dt * self.max_rate);

        spike_sum = np.zeros(10)

        #canvas = Canvas(5, 8)

        for t in range(n_timestep):
            spike_snapshot = np.random.random(self.input_dim) * rescale_fac

            spikes = np.zeros(self.input_dim)
            spikes[spike_snapshot <= x] = 1.
            #canvas.draw(spikes)

            for i, layer in enumerate(self.layers):
                #print "> LAYER %d"%(i + 1)
                spikes = layer.sim(spikes, self.thr)    
                #print "spikes", spikes[:15]
            
            #print
            spike_sum += spikes

        #canvas.show()
        #print spike_sum[:10]

        return np.argmax(spike_sum)


def main():
    optparser = OptionParser()
    optparser.add_option('-n', '--n', dest='n', type='int', default=None)
    optparser.add_option('-e', '--epoch', dest='epoch', type='int', default=30)
    opts, args = optparser.parse_args()

    model_name = 'snn'
    fname_weight = 'model/%s_weights.hdf5'%(model_name)
    fname_config = 'model/%s_config.json'%(model_name)


    f = h5py.File(fname_weight, 'r')

    snn = SNN()
    for i in range(1, 4):
        layer_name = u'dense_%d' % i
        W = f[layer_name][layer_name + '_W']
        snn.add(IFLayer.from_relu(W))

    test = loader.load_test(opts.n)
    imgs, labels = test


    model = model_from_json(open(fname_config, 'r').read())
    model.load_weights(fname_weight)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )

    func_adapt = __import__(model_name, fromlist=['']).__dict__['adapt']
    X, Y = func_adapt(test)
    print model.evaluate(X, Y, verbose=0)[1]
    
    pred = []
    pbar = progbar.start(len(labels))
    for i, img in enumerate(imgs):
        x = img.flatten()
        pred.append(snn.classify(x))
        pbar.update(i + 1)

    pbar.finish()

    pred = np.asarray(pred)
    gold = np.asarray(labels)
    print float(np.sum(pred == gold)) / len(labels)


if __name__ == '__main__':
    main()

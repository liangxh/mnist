#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xihao liang
@created: 2016.11.29
"""


import numpy as np
import math
from optparse import OptionParser
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop

import loader
from optparse import OptionParser

def adapt(XY):
    X, Y = XY
    X = np.asarray(map(lambda x:x.flatten(), X)).astype('float32') / 255
    Y = np_utils.to_categorical(map(int, Y), 10)

    return X, Y

def main():
    optparser = OptionParser()
    optparser.add_option('-n', '--n', dest='n', type='int', default=None)
    optparser.add_option('-e', '--epoch', dest='epoch', type='int', default=30)
    optparser.add_option('-b', '--bias', dest='bias', type='int', default=1)
    opts, args = optparser.parse_args()

    dataset = loader.load()
    gold = dataset[1][1]
    train, test = tuple(map(adapt, dataset))

    model_name = __file__.split('/')[-1].split('.')[0]
    fname_weight = 'model/%s_%d_weights.hdf5'%(model_name, opts.bias)
    fname_config = 'model/%s_%d_config.json'%(model_name, opts.bias)

    bias = (opts.bias == 1)
    model = Sequential()
    model.add(Dense(64, input_dim=784, activation='relu', bias = bias))
    model.add(Dense(128, activation='relu', bias = bias))
    model.add(Dense(10, activation='softmax', bias = bias))

    model.compile(loss='categorical_crossentropy', 
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint(
        fname_weight,
        monitor='acc',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode='max'
    )

    model.fit(
        train[0], train[1],
        batch_size=128,
        nb_epoch=opts.epoch,
        callbacks=[checkpoint,],
        verbose=2,
    )

    model.load_weights(fname_weight)

    loss, acc = model.evaluate(train[0], train[1], verbose=0)
    print 'TRAIN: Loss %.8f Accuracy %.8f'%(loss, acc)

    loss, acc = model.evaluate(test[0], test[1], verbose=0)
    print 'TEST: Loss %.8f Accuracy %.8f'%(loss, acc)


if __name__ == '__main__':
    main()

#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xihao liang
@created: 2016.11.29
"""

import loader

import numpy as np
from optparse import OptionParser
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils


def adapt(XY):
    X, Y = XY
    X = np.asarray(map(lambda x:x.flatten(), X))
    Y = np_utils.to_categorical(map(int, Y), 10)

    return X, Y


def main():
    optparser = OptionParser()
    optparser.add_option('-n', '--n', dest='n', type='int', default=None)
    optparser.add_option('-a', '--activation', dest='activ', default='sigmoid')
    opts, args = optparser.parse_args()

    train, test = tuple(map(adapt, loader.load(opts.n)))
    
    model = Sequential()
    model.add(Dense(64, activation = opts.activ, input_dim=784))
    model.add(Dense(128, activation = opts.activ))
    model.add(Dense(10, activation = 'softmax'))
    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )


    model.fit(
        train[0], train[1],
        batch_size=128,
        nb_epoch=30,
        validation_data=test
    )

    score = model.evaluate(
        test[0], test[1],
        batch_size=128,
    )
    print score

if __name__ == '__main__':
    main()

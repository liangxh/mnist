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
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


def adapt(XY):
    X, Y = XY
    X = np.asarray(map(lambda x:x.reshape(28, 28, 1), X)) / 256
    Y = np_utils.to_categorical(map(int, Y), 10)

    return X, Y


def main():
    optparser = OptionParser()
    optparser.add_option('-n', '--n', dest='n', type='int', default=None)
    optparser.add_option('-e', '--epoch', dest='epoch', type='int', default=30)
    optparser.add_option('-a', '--activation', dest='activ', default='sigmoid')
    optparser.add_option('-b', '--border', dest='border', default='same')
    opts, args = optparser.parse_args()

    model_name = __file__.split('/')[-1].split('.')[0]
    fname_weight = 'model/%s_%s_weights.hdf5'%(model_name, opts.border)
    fname_config = 'model/%s_%s_config.json'%(model_name, opts.border)

    dataset = loader.load(opts.n)
    #test_labels = dataset[-1][1]
    train, test = tuple(map(adapt, dataset))
        
    model = Sequential()
    model.add(Convolution2D(4, 3, 3, border_mode=opts.border, input_shape=(28, 28, 1)))
    model.add(Activation('tanh'))
    #model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2, 2)))  

    model.add(Convolution2D(8, 3, 3, border_mode=opts.border))
    model.add(Activation('tanh'))
    #model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2, 2))) 

    model.add(Flatten())
    model.add(Dense(10, activation = 'softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )

    open(fname_config, 'w').write(model.to_json())

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

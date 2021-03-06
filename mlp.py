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
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, RMSprop


def adapt(XY):
    X, Y = XY
    X = np.asarray(map(lambda x:x.flatten(), X)) / 256
    Y = np_utils.to_categorical(map(int, Y), 10)

    return X, Y


def main():
    optparser = OptionParser()
    optparser.add_option('-n', '--n', dest='n', type='int', default=None)
    optparser.add_option('-e', '--epoch', dest='epoch', type='int', default=30)
    optparser.add_option('-a', '--activation', dest='activ', default='sigmoid')

    opts, args = optparser.parse_args()

    model_name = __file__.split('/')[-1].split('.')[0]
    fname_weight = 'model/%s_%s_weights.hdf5'%(model_name, opts.activ)
    fname_config = 'model/%s_%s_config.json'%(model_name, opts.activ)

    dataset = loader.load(opts.n)
    #test_labels = dataset[-1][1]
    train, test = tuple(map(adapt, dataset))
    
    model = Sequential()
    model.add(Dense(64, activation = opts.activ, input_dim=784))
    model.add(Dense(128, activation = opts.activ))
    model.add(Dense(10, activation = 'softmax'))


    #optimizer = SGD(lr=0.05, momentum=0.0, decay=0.0, nesterov=False)
    #optimizer = RMSprop(lr=0.001)
    optimizer = 'rmsprop'

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
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

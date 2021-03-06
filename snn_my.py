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
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop


def adapt(XY):
    X, Y = XY
    X = np.asarray(map(lambda x:x.flatten(), X)).astype('float32') / 255
    Y = np_utils.to_categorical(map(int, Y), 10)

    return X, Y


def main():
    optparser = OptionParser()
    optparser.add_option('-n', '--n', dest='n', type='int', default=None)
    optparser.add_option('-e', '--epoch', dest='epoch', type='int', default=30)
    opts, args = optparser.parse_args()

    model_name = __file__.split('/')[-1].split('.')[0]
    fname_weight = 'model/%s_weights.hdf5'%(model_name)
    fname_config = 'model/%s_config.json'%(model_name)

    dataset = loader.load(opts.n)
    #test_labels = dataset[-1][1]
    train, test = tuple(map(adapt, dataset))
    
    bias = True
    
    model = Sequential()
    model.add(Dense(64, input_shape=(784,)))
    model.add(Activation('relu'))

    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', 
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

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

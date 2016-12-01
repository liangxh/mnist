#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xihao liang
@created: 2016.11.29
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

import loader

from keras.models import model_from_json

def main():
    if len(sys.argv) < 2:
        print 'model name missing'
        return

    model_name = sys.argv[1]
    func_adapt = __import__(model_name, fromlist=['']).__dict__['adapt']

    fname_weight = 'model/%s_weights.hdf5'%(model_name)
    fname_config = 'model/%s_config.json'%(model_name)

    model = model_from_json(open(fname_config, 'r').read())
    model.load_weights(fname_weight)

    test = loader.load_test()
    imgs = test[0]
    gold = map(int, test[1])

    X, Y = func_adapt(test)
    n_sample = len(Y)
    pred = model.predict_classes(X); print

    print float(np.sum(pred == gold)) / n_sample

    idxs = [i for i in range(n_sample) if not pred[i] == gold[i]]
    for i in range(min(100, len(idxs)))
        

if __name__ == '__main__':
    main()

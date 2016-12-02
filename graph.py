#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

import loader

class Canvas:
    def __init__(self, n_row, n_col):
        self.n_row = n_row
        self.n_col = n_col
        self.count = 0

    def add(self, im, title = None):
        im = im.reshape(28, 28)

        self.count += 1
        if self.count > self.n_row * self.n_col:
            raise Warning("oversize!!")

        plt.subplot(self.n_row, self.n_col, self.count)
        plt.axis('off')
        if title: plt.title(title, fontsize=10)
        plt.imshow(im, cmap='gray')

    def show(self):
        plt.show()

    def savefig(self, fname):
        plt.savefig(fname)

class MnistCanvas:
    def __init__(self, n_row, n_col):
        self.n_row = n_row
        self.n_col = n_col
        self.canvas = np.ones((n_row * 28, n_col * 28)) * 128.

    def add(self, i_row, i_col, im):
        self.canvas[(i_row*28):((i_row + 1)*28), (i_col*28):((i_col + 1)*28)] = im.reshape(28, 28)

    def show(self):
        plt.figure()
        plt.axis('off')
        plt.imshow(self.canvas, cmap='gray')
        plt.show()

    def savefig(self, fname):
        plt.figure()
        plt.axis('off')
        plt.imshow(self.canvas, cmap='gray')
        plt.savefig(fname)

def test():
    imgs, labels = loader.load_test(10)

    #canvas = Canvas(2, 5)
    #for im in imgs: canvas.add(im)

    canvas = MnistCanvas(2, 5)
    for i in range(2):
        for j in range(5):
            canvas.add(i, j, imgs[i * 5 + j])

    canvas.show()


if __name__ == '__main__':
    test()

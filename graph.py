#! /usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import loader


class Canvas:
    def __init__(self, n_row, n_col):
        self.n_row = n_row
        self.n_col = n_col
        self.count = 0    

    def draw(self, im):
        im = im.reshape(28, 28)

        self.count += 1
        if self.count > self.n_row * self.n_col:
            raise Warning("oversize!!")

        plt.subplot(self.n_row, self.n_col, self.count)
        plt.axis('off')
        plt.imshow(im, cmap='gray')

    def show(self):
        plt.show()


def test():
    imgs, labels = loader.load_test(10)

    canvas = Canvas(2, 5)
    for im in imgs: canvas.draw(im)

    canvas.show()


if __name__ == '__main__':
    test()

#! /bin/bash

if [ ! -d data ]; then
    mkdir data
fi

cd data
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

for f in $(ls | grep '\.gz'); do
    tar -zxvf $f
    rm $f
done

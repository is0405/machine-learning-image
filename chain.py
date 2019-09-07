# -*- coding: utf-8 -*-

import math
import random
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
# chainer
from chainer import Chain, Variable, iterators, training, datasets, serializers
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from Chainer import Chainer
from chainer.training import extensions, triggers

from chainer.datasets import TransformDataset
from chainer.datasets import LabeledImageDataset

import os
import glob
from itertools import chain
from PIL import Image

width, height = 224, 224 #ここは好きなサイズで構いません。
# 各データに行う変換
def transform(data):
    img, label = data
    img = img.astype(np.uint8)
    img = Image.fromarray(img.transpose(1, 2, 0))
    img = img.resize((width, height))
    img = np.asarray(img).transpose(2, 0, 1).astype(np.float32) / 255.
    return img, label

if __name__ == '__main__':
    #画像の前処理
    train = LabeledImageDataset('data/train/train_labels.txt', root = 'data/train/images')
    train = TransformDataset(train, transform)

    valid = LabeledImageDataset('data/valid/valid_labels.txt', root = 'data/valid/images')
    valid = TransformDataset(valid, transform)
    
    epoch = 5
    batch = 9

    model = L.Classifier(Chainer())
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    train_iter = iterators.SerialIterator(train, batch)
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out='result')
    
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.PlotReport(['main/loss'], 'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy'], 'epoch', file_name='accuracy.png'))
    trainer.run()

    serializers.save_npz("mymodel.npz", model)


















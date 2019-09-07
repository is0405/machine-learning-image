# -*- coding: utf-8 -*-
from chainer import Chain
import chainer.links as L
import chainer.functions as F

class Chainer(Chain):

    def __init__(self):
        super(Chainer, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, 3, pad=1)
            self.conv2 = L.Convolution2D(None, 32, 3, pad=1)
            self.l3 = L.Linear(None, 256)
            self.l4 = L.Linear(None, 9)

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=3, stride=2, pad=1)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=3, stride=2, pad=1)
        h = F.dropout(F.relu(self.l3(h)))
        y = self.l4(h)
        return y

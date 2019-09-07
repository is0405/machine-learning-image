import os
import shutil
import glob
import random
import re
import chainer
import numpy as np
from itertools import chain
from chainer import iterators, training, optimizers, datasets, serializers
from chainer.dataset import concat_examples
import chainer.functions as F
import chainer.links as L
from Chainer import Chainer
from PIL import Image

chainer.config.train = False
width, height = 224, 224 #ここは好きなサイズで構いません。

# モデルの読み込み
model = L.Classifier(Chainer())
serializers.load_npz("mymodel.npz", model)

def choice_images():
    reportfiles = [r.split('/')[-1] for r in glob.glob('data/test/images/*.jpg')]
    posttext = random.choice(reportfiles)
    return posttext

def search_images(image):
    path = 'data/test/test_labels.txt'
    with open(path) as f:
        for i, line in enumerate(f):
            if image in line:
                break
        return line.split()

def transform(image):
    img = image.astype(np.uint8).transpose(2, 1, 0)
    img = Image.fromarray(img.transpose(1, 2, 0))
    img = img.resize((width, height))
    img = np.asarray(img).transpose(2, 0, 1).astype(np.float32) / 255.
    return img

image = choice_images()
ans = search_images(image)
print(ans)

image = np.array(Image.open('data/test/images/' + image))
img_arr = transform(image)

t = model.predictor(img_arr[None, ...])
print(int(t.data.argmax()))

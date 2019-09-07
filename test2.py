import os
import glob
import chainer
import numpy as np
from itertools import chain
from chainer import iterators, training, optimizers, datasets, serializers
from chainer.dataset import concat_examples
import chainer.functions as F
import chainer.links as L
from Chainer import Chainer
from PIL import Image
from tqdm import tqdm

chainer.config.train = False
width, height = 224, 224 #ここは好きなサイズで構いません。

# モデルの読み込み
model = L.Classifier(Chainer())
serializers.load_npz("mymodel.npz", model)

def search_images(image):
    path_t = 'data/test/test_labels.txt'
    with open(path_t) as f:
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

path = glob.glob('data/test/images/*.jpg')
count_t = 0

for i in tqdm(range(len(path))):
    image = path[i]
    ans = search_images(image)
    collect_num = int(ans[1])

    image = np.array(Image.open(image))
    img_arr = transform(image)

    t = model.predictor(img_arr[None, ...])
    predict_num = int(t.data.argmax())

    if collect_num == predict_num:
        count_t += 1

print("試験回数 " + str(len(path)))
print("正答数 " + str(count_t))
print("正答率は" + str(count_t / len(path):.2f) + "%です")

import json
import os
import cv2
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import logging
import pandas as pd
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

from data.data_process import DataProcessor
from keras_cls.train import Image_Classification_Parameter
from keras_match.match_net import MatchNet
from keras_cls.utils.preprocess import normalize, resize_img

def test_match_net():
    match_net = MatchNet()

    root_path = r'/data/math_research/test_paper_cls/dataset/train/'
    image_file_list = (r'paper_20220506_2_2.jpg', r'paper_20220506_2_4.jpg')
    image_list = []
    for image_file in image_file_list:
        image_path = os.path.join(root_path, image_file)
        image = np.ascontiguousarray(Image.open(image_path).convert('RGB'))
        image, _, _ = resize_img(image, match_net.img_cls_params.progressive_resizing[0])
        # image = self.baseline.distort(image)
        # self.show_img(image_path, image)
        image = normalize(image, mode='tf')
        image_list.append(image)
    model = match_net.get_match_net()
    print(model.summary())
    pred_result = model([image_list[0], image_list[1]])
    pred_cls = np.argmax(pred_result, axis=-1)
    print(pred_cls)

    print("hello")

if __name__ == '__main__':
    test_match_net()
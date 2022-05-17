import os, argparse, sys
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from sklearn import metrics as sklearn_metrics
import time
from tqdm import tqdm
from tensorflow.keras.layers import Concatenate, Flatten, Dense, Activation
from tensorflow.keras import Model
from tensorflow.keras.activations import softmax, sigmoid

from keras_cls.model.model_builder import get_model
from keras_cls.train import Image_Classification_Parameter
from keras_cls.utils.class_head import class_head
from keras_cls.utils.common import get_best_model_path
from keras_cls.utils.preprocess import normalize, resize_img

class MatchNet:
    def __init__(self):
        self.img_cls_params = Image_Classification_Parameter()
        self.img_cls_params.num_classes = 2
        self.img_cls_params.backbone = r'EfficientNetB5'
        self.img_cls_params.progressive_resizing = [(456, 456)]
        self.img_cls_params.epochs = 100
        self.img_cls_params.batch_size = 8

    def get_input_model(self):
        model = get_model(self.img_cls_params,
                          num_class=self.img_cls_params.num_classes,
                          include_top=False)
        return model

    def get_match_net(self):
        image_left = self.get_input_model()
        image_right = self.get_input_model()
        # fci = merge([input1.output, input2.output])
        fci = Concatenate()([image_left.output, image_right.output])
        fc0 = Flatten()(fci)
        fc1 = Dense(1024, activation='relu')(fc0)
        fc2 = Dense(1024, activation='relu')(fc1)
        fc2 = class_head(fc2, self.img_cls_params.num_classes, 512, dropout=self.img_cls_params.dropout)
        if self.img_cls_params.loss == 'ce' or self.img_cls_params.num_classes > 2:
            fc3 = Activation(softmax, dtype='float32', name="predictions")(fc2)
        else:
            fc3 = Activation(sigmoid, dtype='float32', name="predictions")(fc2)
        models = Model(inputs=[image_left.input, image_right.input], outputs=fc3)
        return models

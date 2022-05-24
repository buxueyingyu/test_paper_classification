import os, argparse, sys
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from sklearn import metrics as sklearn_metrics
import time
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Convolution2D, Conv2D, MaxPool2D, \
    Flatten, Dense, Input, Lambda, Dropout, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras import Model, Sequential
from tensorflow.keras.activations import softmax, sigmoid
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

from keras_cls.model.model_builder import get_model
from keras_cls.train import Image_Classification_Parameter
from keras_cls.utils.class_head import class_head
from keras_cls.utils.common import get_best_model_path
from keras_cls.utils.preprocess import normalize, resize_img
from keras_cls.utils.common import set_mixed_precision, get_confusion_matrix
from keras_match.generator.generator_builder import get_generator
from keras_cls.losses.losses_builder import get_losses
from keras_cls.optimizer.optimizer_builder import get_optimizer
from keras_cls.utils.lr_finder import LRFinder
from keras_cls.utils.lr_scheduler import get_lr_scheduler
from keras_match.model.loss import euclidean_distance, eucl_dist_output_shape, contrastive_loss


class SimpleMatchNet:
    def __init__(self,
                 img_cls_params: Image_Classification_Parameter,
                 with_simple_network: bool = True,
                 last_dense_size: bool = 4096,
                 single_backbone: bool = False):
        self.img_cls_params = img_cls_params
        self.img_cls_params.init_lr = 10e-4
        self.with_simple_network = with_simple_network
        self.last_dense_size = last_dense_size
        self.single_backbone = single_backbone
        base_size = self.img_cls_params.progressive_resizing[0]
        self.input_shape = (base_size[0], base_size[1], 3)
        self.l2_penalization = {'Conv1': 1e-2,
                                'Conv2': 1e-2,
                                'Conv3': 1e-2,
                                'Conv4': 1e-2,
                                'Dense1': 1e-4}

    def get_simple_network_v2(self):
        '''Base network to be shared (eq. to feature extraction).
        '''
        seq = Sequential()
        nb_filter = [6, 12]
        kern_size = 3
        # conv layers
        # seq.add(Reshape((1, 38, 31), input_shape=(38, 31)))
        seq.add(Conv2D(filters=nb_filter[0],
                       activation='relu',
                       kernel_size=(kern_size, kern_size),
                       input_shape=self.input_shape))
        seq.add(MaxPool2D(pool_size=(2, 2)))  # downsample
        seq.add(Dropout(.25))
        # conv layer 2
        seq.add(Conv2D(filters=nb_filter[1],
                       activation='relu',
                       kernel_size=(kern_size, kern_size),
                       input_shape=self.input_shape))
        seq.add(MaxPool2D(pool_size=(2, 2)))  # downsample
        seq.add(Dropout(.25))

        # dense layers
        seq.add(Flatten())
        seq.add(Dense(128, activation='relu'))
        seq.add(Dropout(0.1))
        seq.add(Dense(50, activation='relu'))
        return seq

    def get_simple_network(self):
        convolutional_net = Sequential()
        convolutional_net.add(Conv2D(filters=16, kernel_size=(10, 10),
                                     activation='relu',
                                     input_shape=self.input_shape,
                                     kernel_regularizer=l2(
                                         self.l2_penalization['Conv1']),
                                     name='Conv1'))
        convolutional_net.add(MaxPool2D())
        convolutional_net.add(Dropout(0.1))
        convolutional_net.add(Conv2D(filters=32, kernel_size=(7, 7),
                                     activation='relu',
                                     kernel_regularizer=l2(
                                         self.l2_penalization['Conv2']),
                                     name='Conv2'))
        convolutional_net.add(MaxPool2D())
        convolutional_net.add(Dropout(0.1))
        convolutional_net.add(Conv2D(filters=64, kernel_size=(4, 4),
                                     activation='relu',
                                     kernel_regularizer=l2(
                                         self.l2_penalization['Conv3']),
                                     name='Conv3'))
        convolutional_net.add(MaxPool2D())
        convolutional_net.add(Dropout(0.1))
        convolutional_net.add(Conv2D(filters=128, kernel_size=(4, 4),
                                     activation='relu',
                                     kernel_regularizer=l2(
                                         self.l2_penalization['Conv4']),
                                     name='Conv4'))

        convolutional_net.add(Flatten())

        convolutional_net.add(
            Dense(units=self.last_dense_size, activation='sigmoid',
                  kernel_regularizer=l2(
                      self.l2_penalization['Dense1']),
                  name='Dense1'))
        return convolutional_net

    def get_efficientnet_network(self):
        model = get_model(self.img_cls_params,
                          num_class=self.img_cls_params.num_classes,
                          include_top=False)
        return model

    def get_match_net(self):
        """
        直接将欧式距离进行输出
        Returns:

        """
        if self.with_simple_network:
            backbone_model = self.get_simple_network()
        else:
            backbone_model = self.get_efficientnet_network()
        left_input = Input(shape=self.input_shape, name='left_input')
        right_input = Input(shape=self.input_shape, name='right_input')
        left_output = backbone_model(left_input)
        right_output = backbone_model(right_input)

        distance = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape)([left_output, right_output])

        models = Model(inputs=[left_input, right_input],
                       outputs=distance,
                       name='double_tower_single_backbone')
        return models

    def train(self):
        assert self.img_cls_params.label_file and os.path.exists(self.img_cls_params.label_file)
        assert self.img_cls_params.dataset_dir and os.path.exists(self.img_cls_params.dataset_dir)
        # set mixed_precision to float16
        if self.img_cls_params.mixed_precision:
            set_mixed_precision('mixed_float16')
        # build dataset and model
        train_generator, val_generator = get_generator(self.img_cls_params, is_grey=True)
        # perform data sanity check to identify invalid inputs
        # show_training_images(train_generator, num_img=9)
        # check class imbalance
        # show_classes_hist(train_generator.class_counts, train_generator.class_names)

        model = self.get_match_net()
        print(model.summary())
        """
        重点：使用contrastive_loss进行loss计算
        """
        loss_fn = contrastive_loss
        optimizer = get_optimizer(self.img_cls_params)

        # create directory to save checkpoints
        os.makedirs(self.img_cls_params.checkpoints, exist_ok=True)

        print("loading dataset...")
        start_time = time.perf_counter()
        best_val_loss = np.inf
        best_val_epoch = -1
        # train_writer = tf.summary.create_file_writer("logs/train_loss")
        # val_writer = tf.summary.create_file_writer("logs/val_loss")
        # acc_writer = tf.summary.create_file_writer("logs/acc")

        # lr finder
        if self.img_cls_params.init_lr == 0:
            print("\nlr finder is running...")
            lr_finder = LRFinder(start_lr=1e-7, end_lr=1.0, num_it=max(min(len(train_generator) // 2, 100), 30))
            lr_finder.find_lr(train_generator, model, loss_fn, optimizer, self.img_cls_params)
            # show training loss
            lr_finder.plot_loss()
            best_init_lr = lr_finder.get_best_lr()
            print("\nbest_init_lr:{}".format(best_init_lr))
            self.img_cls_params.init_lr = best_init_lr

        # training
        for epoch in range(int(self.img_cls_params.epochs)):
            lr = get_lr_scheduler(self.img_cls_params)(epoch)
            optimizer.learning_rate.assign(lr)
            remaining_epoches = self.img_cls_params.epochs - epoch - 1
            epoch_start_time = time.perf_counter()

            if self.img_cls_params.progressive_resizing:
                img_size_index = int(epoch //
                                     np.ceil(self.img_cls_params.epochs /
                                             len(self.img_cls_params.progressive_resizing)))
                img_size = self.img_cls_params.progressive_resizing[img_size_index]
                train_generator.set_img_size(img_size)
                print("progressive resizing:", img_size)
            train_loss = 0
            train_generator_tqdm = tqdm(enumerate(train_generator), total=len(train_generator), ncols=200)
            for batch_index, (batch_imgs, batch_labels) in train_generator_tqdm:
                with tf.GradientTape() as tape:
                    model_outputs = model([batch_imgs[:, 0], batch_imgs[:, 1]], training=True)
                    data_loss = loss_fn(batch_labels, model_outputs)
                    total_loss = data_loss + self.img_cls_params.weight_decay * tf.add_n(
                        [tf.nn.l2_loss(v) for v in model.trainable_variables if '_bn' not in v.name])
                grads = tape.gradient(total_loss, model.trainable_variables)
                if self.img_cls_params.optimizer.startswith('SAM'):
                    optimizer.first_step(grads, model.trainable_variables)
                    with tf.GradientTape() as tape:
                        model_outputs = model([batch_imgs[:, 0], batch_imgs[:, 1]], training=True)
                        data_loss = loss_fn(batch_labels, model_outputs)
                        total_loss = data_loss + self.img_cls_params.weight_decay * tf.add_n(
                            [tf.nn.l2_loss(v) for v in model.trainable_variables if '_bn' not in v.name])
                    grads = tape.gradient(total_loss, model.trainable_variables)
                    optimizer.second_step(grads, model.trainable_variables)
                else:
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                train_loss += data_loss
                train_generator_tqdm.set_description(
                    "epoch:{}/{},train_loss:{:.5f},lr:{:.6f}".format(epoch + 1, self.img_cls_params.epochs,
                                                                     train_loss / (batch_index + 1),
                                                                     optimizer.learning_rate.numpy()))
            train_generator.on_epoch_end()

            # evaluation
            if epoch >= self.img_cls_params.start_eval_epoch - 1:
                val_loss = 0
                val_acc = 0
                num_img = 0
                cur_val_loss = cur_val_acc = 0
                val_generator_tqdm = tqdm(enumerate(val_generator), total=len(val_generator), ncols=200)
                for batch_index, (batch_imgs, batch_labels) in val_generator_tqdm:
                    model_outputs = model([batch_imgs[:, 0], batch_imgs[:, 1]])
                    total_loss = loss_fn(batch_labels, model_outputs)
                    val_loss += total_loss
                    # 欧式距离如果小于0.5，则认为图片极其相似
                    wrong_pred_mask = (batch_labels == tf.less(model_outputs, 0.5))
                    val_acc += np.sum(wrong_pred_mask)
                    num_img += np.shape(batch_labels)[0]
                    cur_val_loss = val_loss / num_img
                    cur_val_acc = val_acc / num_img
                    val_generator_tqdm.set_description(
                        "epoch:{}/{},val_loss:{:.5f},val_acc:{:.5f}".format(epoch + 1,
                                                                            self.img_cls_params.epochs,
                                                                            cur_val_loss,
                                                                            cur_val_acc))
                if cur_val_loss < best_val_loss:
                    best_val_loss = cur_val_loss
                    best_val_epoch = epoch + 1
                    if self.img_cls_params.is_simple_network:
                        backbone = 'simple'
                    else:
                        backbone = self.img_cls_params.backbone
                    best_weight_path = os.path.join(self.img_cls_params.checkpoints,
                                                    'img_cls_best_weight_{}_val_loss_{:.3f}_val_acc_{:.3f}_epoch_{}'
                                                    .format(backbone,
                                                            best_val_loss,
                                                            cur_val_acc,
                                                            best_val_epoch))
                    model.save_weights(best_weight_path)

            cur_time = time.perf_counter()
            one_epoch_time = cur_time - epoch_start_time
            print("time elapsed: {:.3f} hour, time left: {:.3f} hour"
                  .format((cur_time - start_time) / 3600,
                          remaining_epoches * one_epoch_time / 3600))
        print("finished!")

    def validate_train_model(self):
        try:
            # set mixed_precision to float16
            if self.img_cls_params.mixed_precision:
                set_mixed_precision('mixed_float16')
            # build dataset and model
            train_generator, val_generator = get_generator(self.img_cls_params, is_grey=True)
            # show prediction result
            model = self.get_match_net()
            model_path = get_best_model_path(self.img_cls_params.checkpoints)
            model.load_weights(model_path)
            wrong_pred_result = get_confusion_matrix(val_generator, model)
            print("wrong_pred_result:{}".format(wrong_pred_result))
        except:
            pass

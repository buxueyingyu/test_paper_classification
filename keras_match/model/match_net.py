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
from tensorflow.keras.layers import Concatenate, Flatten, Dense, Activation
from tensorflow.keras import Model
from tensorflow.keras.activations import softmax, sigmoid

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


class MatchNet:
    def __init__(self, img_cls_params: Image_Classification_Parameter):
        self.img_cls_params = img_cls_params

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

    def train_image_classification_model(self):
        assert self.img_cls_params.label_file and os.path.exists(self.img_cls_params.label_file)
        assert self.img_cls_params.dataset_dir and os.path.exists(self.img_cls_params.dataset_dir)
        # set mixed_precision to float16
        if self.img_cls_params.mixed_precision:
            set_mixed_precision('mixed_float16')
        # build dataset and model
        train_generator, val_generator = get_generator(self.img_cls_params)
        # perform data sanity check to identify invalid inputs
        # show_training_images(train_generator, num_img=9)
        # check class imbalance
        # show_classes_hist(train_generator.class_counts, train_generator.class_names)

        model = self.get_match_net()
        loss_fn = get_losses(self.img_cls_params)
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
            # lr_finder.plot_loss_change()
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
            train_generator_tqdm = tqdm(enumerate(train_generator), total=len(train_generator))
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

            # with train_writer.as_default():
            #     tf.summary.scalar("train_loss-val_loss", train_loss / (len(train_generator) * args.batch_size), step=epoch)
            #     train_writer.flush()
            # evaluation
            if epoch >= self.img_cls_params.start_eval_epoch - 1:
                val_loss = 0
                val_acc = 0
                num_img = 0
                cur_val_loss = cur_val_acc = 0
                val_generator_tqdm = tqdm(enumerate(val_generator), total=len(val_generator))
                for batch_index, (batch_imgs, batch_labels) in val_generator_tqdm:
                    model_outputs = model([batch_imgs[:, 0], batch_imgs[:, 1]])
                    total_loss = loss_fn(batch_labels, model_outputs)
                    val_loss += total_loss
                    wrong_pred_mask = np.argmax(batch_labels, axis=-1) == np.argmax(model_outputs, axis=-1)
                    val_acc += np.sum(wrong_pred_mask)
                    num_img += np.shape(batch_labels)[0]
                    cur_val_loss = val_loss / num_img
                    cur_val_acc = val_acc / num_img
                    val_generator_tqdm.set_description(
                        "epoch:{}/{},val_loss:{:.5f},val_acc:{:.5f}".format(epoch + 1,
                                                                            self.img_cls_params.epochs,
                                                                            cur_val_loss,
                                                                            cur_val_acc))
                # with val_writer.as_default():
                #     tf.summary.scalar("train_loss-val_loss", cur_val_loss, step=epoch)
                #     val_writer.flush()
                # with acc_writer.as_default():
                #     tf.summary.scalar("acc", cur_val_acc, step=epoch)
                #     acc_writer.flush()
                if cur_val_loss < best_val_loss:
                    best_val_loss = cur_val_loss
                    best_val_epoch = epoch + 1
                    best_weight_path = os.path.join(self.img_cls_params.checkpoints,
                                                    'img_cls_best_weight_{}_val_loss_{:.3f}_val_acc_{:.3f}_epoch_{}'
                                                    .format(self.img_cls_params.backbone,
                                                            best_val_loss,
                                                            cur_val_acc,
                                                            best_val_epoch))
                    model.save_weights(best_weight_path)

            cur_time = time.perf_counter()
            one_epoch_time = cur_time - epoch_start_time
            print("time elapsed: {:.3f} hour, time left: {:.3f} hour"
                  .format((cur_time - start_time) / 3600,
                          remaining_epoches * one_epoch_time / 3600))

            # if epoch >= 1 and not open_tensorboard_url:
            #     open_tensorboard_url = True
            #     webbrowser.open(url, new=1)
        # try:
        #     # show prediction result
        #     model_path = get_best_model_path(args.checkpoints)
        #     model.load_weights(model_path)
        #     wrong_pred_result = get_confusion_matrix(val_generator, model)
        #     print("wrong_pred_result:{}".format(wrong_pred_result))
        # except:
        #     pass
        print("finished!")

    def validate_train_model(self):
        try:
            # set mixed_precision to float16
            if self.img_cls_params.mixed_precision:
                set_mixed_precision('mixed_float16')
            # build dataset and model
            train_generator, val_generator = get_generator(self.img_cls_params)
            # show prediction result
            model = self.get_match_net()
            model_path = get_best_model_path(self.img_cls_params.checkpoints)
            model.load_weights(model_path)
            wrong_pred_result = get_confusion_matrix(val_generator, model)
            print("wrong_pred_result:{}".format(wrong_pred_result))
        except:
            pass

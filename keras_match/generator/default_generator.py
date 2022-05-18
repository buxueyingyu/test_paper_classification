import numpy as np
import os
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import albumentations as A
import random
import pandas as pd
from keras_cls.utils.preprocess import normalize
from keras_cls.augment import auto_augment, rand_augment, \
    baseline_augment, mixup, cutmix
from keras_cls.utils.preprocess import resize_img_aug, resize_img


class DefaultGenerator(tf.keras.utils.Sequence):
    def __init__(self, args, mode='train', train_valid_split_ratio=0.7, dataset_sample_ratio=1.0):
        self.args = args
        self.dataset_dir = args.dataset_dir
        batch_size = args.batch_size
        augment = args.augment

        self.class_counts = []
        self.train_valid_split_ratio = train_valid_split_ratio
        random.seed(123)
        # np.random.seed(123)
        self.set_img_size(args.progressive_resizing[-1])

        (train_img_path_list, val_img_path_list, train_label_list, val_label_list, class_names) = \
            self.create_split_list(self.dataset_dir,
                                   args.checkpoints,
                                   args.label_file,
                                   dataset_sample_ratio,
                                   mode)
        # print(train_img_path_list)
        # print(val_img_path_list)
        if mode == 'train':
            self.img_path_list = train_img_path_list
            self.label_list = train_label_list
            img_path_list_len = len(self.img_path_list)
            pad_len = batch_size - img_path_list_len % batch_size
            pad_img_path_list = []
            pad_label_list = []
            for _ in range(pad_len):
                rand_index = np.random.randint(0, img_path_list_len)
                pad_img_path_list.append(self.img_path_list[rand_index])
                pad_label_list.append(self.label_list[rand_index])

            self.img_path_list = np.append(self.img_path_list, pad_img_path_list, axis=0)
            self.label_list = np.append(self.label_list, pad_label_list)
            self.data_index = np.arange(0, len(self.label_list))
            np.random.shuffle(self.data_index)
        else:
            self.img_path_list = val_img_path_list
            self.label_list = val_label_list
            self.data_index = np.arange(0, len(self.label_list))
            self.valid_mask = [True] * len(val_label_list)
        self.img_path_list = np.array(self.img_path_list)
        self.label_list = np.array(self.label_list)
        self.augment = augment
        self.mode = mode
        self.batch_size = batch_size
        self.eppch_index = 0
        self.class_names = class_names
        self.num_class = len(class_names)

        self.auto_augment = auto_augment.AutoAugment(augmentation_name='v0')
        self.rand_augment = rand_augment.RandAugment(num_layers=2, magnitude=10.)
        self.mixup = mixup.Mixup(beta=1, prob=1.)
        self.cutmix = cutmix.Cutmix(beta=1, prob=1.)
        self.baseline_augment = baseline_augment.Baseline()

    def read_img(self, path):
        image = np.ascontiguousarray(Image.open(path).convert('RGB'))
        return image
        # return image[:, :, ::-1]

    def on_epoch_end(self):
        if self.mode == 'train':
            np.random.shuffle(self.data_index)
        # else:
        #     self.valid_mask=[True]*len(self.label_list)
        self.eppch_index += 1

    def __len__(self):
        return int(np.ceil(len(self.img_path_list) / self.batch_size))

    def __getitem__(self, batch_index):
        batch_img_paths = self.img_path_list[
            self.data_index[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]]
        batch_labels = self.label_list[
            self.data_index[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]]
        one_hot_batch_labels = np.zeros([len(batch_img_paths), self.num_class])
        batch_imgs = []
        if self.mode == "valid":
            valid_index = 0
            for i in range(len(batch_img_paths)):
                try:
                    left_img_path, right_img_path = batch_img_paths[i]
                    left_img = self.read_img(left_img_path)
                    right_img = self.read_img(right_img_path)
                except:
                    self.valid_mask[batch_index * self.batch_size + i] = False
                    continue

                left_img = self.valid_resize_img(left_img)
                right_img = self.valid_resize_img(right_img)
                if self.args.weights:
                    if self.args.backbone[0:3] == "Res":
                        left_img = normalize(left_img, mode='caffe')
                        right_img = normalize(right_img, mode='caffe')
                    else:
                        left_img = normalize(left_img, mode='tf')
                        right_img = normalize(right_img, mode='tf')
                else:
                    left_img = normalize(left_img, mode='tf')
                    right_img = normalize(right_img, mode='tf')

                batch_imgs.append((left_img, right_img))
                one_hot_batch_labels[valid_index, batch_labels[i]] = 1
                valid_index += 1
            batch_imgs = np.array(batch_imgs)
            one_hot_batch_labels = one_hot_batch_labels[:valid_index]
        else:
            valid_index = 0
            for i in range(len(batch_img_paths)):
                try:
                    left_img_path, right_img_path = batch_img_paths[i]
                    left_img = self.read_img(left_img_path).astype(np.uint8)
                    right_img = self.read_img(right_img_path).astype(np.uint8)
                except:
                    continue

                left_img = self.train_resize_img(left_img)
                right_img = self.train_resize_img(right_img)

                if self.augment == 'auto_augment':
                    left_img = self.auto_augment.distort(tf.constant(left_img)).numpy()
                    right_img = self.auto_augment.distort(tf.constant(right_img)).numpy()
                elif self.augment == 'rand_augment':
                    left_img = self.rand_augment.distort(tf.constant(left_img)).numpy()
                    right_img = self.rand_augment.distort(tf.constant(right_img)).numpy()
                elif self.augment == 'baseline':
                    left_img = self.baseline_augment.distort(left_img)
                    right_img = self.baseline_augment.distort(right_img)
                if self.args.weights:
                    if self.args.backbone[0:3] == "Res":
                        left_img = normalize(left_img, mode='caffe')
                        right_img = normalize(right_img, mode='caffe')
                    else:
                        left_img = normalize(left_img, mode='tf')
                        right_img = normalize(right_img, mode='tf')
                else:
                    left_img = normalize(left_img, mode='tf')
                    right_img = normalize(right_img, mode='tf')
                batch_imgs.append((left_img, right_img))
                one_hot_batch_labels[valid_index, batch_labels[i]] = 1
                valid_index += 1
            batch_imgs = np.array(batch_imgs)
            one_hot_batch_labels = one_hot_batch_labels[:valid_index]

        return batch_imgs, one_hot_batch_labels

    def create_split_list(self, img_dir, model_dir, label_file: str, dataset_sample_ratio, mode):
        train_imgs_list = []
        val_imgs_list = []
        train_labels_list = []
        val_labels_list = []
        label_data = pd.read_csv(label_file, encoding='utf-8', sep=',')
        valid_class_names = sorted(list(set(label_data['label'])))
        for class_index, class_name in enumerate(valid_class_names):
            class_data = label_data[label_data['label'] == class_name]
            file_list = []
            for index, row in class_data.iterrows():
                left_img_path = os.path.join(img_dir, row['left'])
                right_img_path = os.path.join(img_dir, row['right'])
                if os.path.exists(left_img_path) and os.path.exists(right_img_path):
                    file_list.append((left_img_path, right_img_path))

            num_file = int(len(file_list) * dataset_sample_ratio)
            file_list = random.sample(file_list, num_file)
            num_train = int(num_file * self.train_valid_split_ratio)
            train_imgs_list.extend(file_list[:num_train])
            val_imgs_list.extend(file_list[num_train:])
            train_labels_list.extend([class_index] * num_train)
            val_labels_list.extend([class_index] * (num_file - num_train))
            self.class_counts.append(num_file)

        train_imgs_list = np.array(train_imgs_list)
        train_labels_list = np.array(train_labels_list)
        val_imgs_list = np.array(val_imgs_list)
        val_labels_list = np.array(val_labels_list)

        random_index = random.sample(range(len(train_imgs_list)), len(train_imgs_list))
        train_imgs_list = train_imgs_list[random_index]
        train_labels_list = train_labels_list[random_index]
        random_index = random.sample(range(len(val_imgs_list)), len(val_imgs_list))
        val_imgs_list = val_imgs_list[random_index]
        val_labels_list = val_labels_list[random_index]

        if mode == 'train':
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, 'class.names'), 'w') as f1:
                for val in valid_class_names:
                    f1.write(str(val) + "\n")
        return train_imgs_list, val_imgs_list, train_labels_list, val_labels_list, valid_class_names

    def valid_resize_img(self, img):
        dst_size = self.resize_size
        img, _, _ = resize_img(img, dst_size)
        img = self.center_crop_transform(image=img)['image']
        return img

    def train_resize_img(self, img):
        dst_size = self.resize_size
        img, _, _ = resize_img_aug(img, dst_size)
        img = self.random_crop_transform(image=img)['image']
        return img

    def set_img_size(self, img_size):
        img_w = img_size[0]
        img_h = img_size[1]
        self.resize_size = (int(img_w / 0.875), int(img_h / 0.875))
        self.crop_size = (img_w, img_h)
        self.random_crop_transform = A.Compose([
            A.RandomCrop(width=self.crop_size[0], height=self.crop_size[1]),
        ])
        self.center_crop_transform = A.Compose([
            A.CenterCrop(width=self.crop_size[0], height=self.crop_size[1]),
        ])

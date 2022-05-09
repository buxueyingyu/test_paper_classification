import os
import logging
import shutil
import time
from tqdm import tqdm
from copy import deepcopy
from albumentations import (
    CLAHE, Crop,
    RandomRotate90, Rotate, Transpose, ShiftScaleRotate, Blur,
    OpticalDistortion, GridDistortion, HueSaturationValue,
    GaussNoise, MotionBlur, MedianBlur,
    RandomContrast,
    RandomBrightness, Flip, OneOf, Compose, ImageCompression
)
import cv2
import random
from sklearn.model_selection import train_test_split
from glob import glob
import pandas as pd

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self):
        pass

    def rename_files(self,
                     data_path: str):
        if not data_path or not isinstance(data_path, str) or not os.path.exists(data_path):
            return

        header = 'paper'
        middle = time.strftime('%Y%m%d', time.localtime(time.time()))
        file_list = os.listdir(data_path)
        file_list = [file for file in file_list if not file.startswith('paper')]
        proper_index = None
        for index, file in enumerate(tqdm(file_list)):
            if proper_index is None:
                proper_index = index
            old_path = os.path.join(data_path, file)
            new_path, proper_index = self.get_non_exist_name(data_path, header, middle, proper_index)
            proper_index += 1
            os.rename(old_path, new_path)
            logger.info(f'{file} --> {os.path.basename(new_path)}')

    def get_non_exist_name(self, root_path: str, header: str, middle: str, current_index: int):
        compare_path = os.path.join(root_path, f'{header}_{middle}_{current_index}.jpg')
        while os.path.exists(compare_path):
            current_index += 1
            new_name = f'{header}_{middle}_{current_index}.jpg'
            compare_path = os.path.join(root_path, new_name)
        return compare_path, current_index

    def light_aug(self, image):
        original_height, original_width = image.shape[:2]
        compose = Compose([Crop(x_min=int(original_width * random.uniform(0, 0.15)),
                                y_min=int(original_height * random.uniform(0, 0.15)),
                                x_max=int(original_width * random.uniform(0.9, 1)),
                                y_max=int(original_height * random.uniform(0.9, 1)),
                                p=0.5)], p=1)
        return compose(image=image)['image']

    def augment_by_file(self, image_file, p=1):
        if not image_file or not isinstance(image_file, str) or not os.path.exists(image_file):
            return None
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.augmentation(image, p)

    def augment(self, image, p=1):
        original_height, original_width = image.shape[:2]
        compose = Compose([
            Crop(x_min=int(original_width * random.uniform(0, 0.15)),
                 y_min=int(original_height * random.uniform(0, 0.15)),
                 x_max=int(original_width * random.uniform(0.9, 1)),
                 y_max=int(original_height * random.uniform(0.9, 1)),
                 p=0.3),
            RandomRotate90(),
            Flip(p=0.4),
            Rotate(p=0.4),
            Transpose(p=0.3),
            GaussNoise(p=0.2),
            OneOf([
                MotionBlur(p=.5),
                MedianBlur(blur_limit=3, p=.5),
                Blur(blur_limit=3, p=.5),
            ], p=0.4),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
            OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=.3)
            ], p=0.3),
            CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1.0),
            OneOf([
                RandomContrast(),
                RandomBrightness()
            ], p=0.3),
            HueSaturationValue(p=0.3),
            ImageCompression(quality_lower=int(100 * random.uniform(0.5, 0.8)), quality_upper=100, p=0.5)
        ], p=p)
        image_aug = compose(image=image)['image']
        return image_aug

    def output_augment_image_file(self,
                                  image_file: str,
                                  output_path: str,
                                  augment_amounts: int = 20):
        if not image_file or not isinstance(image_file, str) or not os.path.exists(image_file):
            return None
        if not output_path and not isinstance(output_path, str) or len(output_path) == 0:
            return None
        if not isinstance(augment_amounts, int) or augment_amounts < 1:
            return None
        os.makedirs(output_path, exist_ok=True)
        pure_name = os.path.basename(image_file).split('.')[0]
        logger.info(f'Augment image: {image_file}')
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        new_file_name = f'{pure_name}_origin.jpg'
        new_file_path = os.path.join(output_path, new_file_name)
        cv2.imwrite(new_file_path, image, [cv2.IMWRITE_JPEG_QUALITY, 50])
        for index in range(augment_amounts):
            new_file_name = f'{pure_name}_{index}.jpg'
            new_file_path = os.path.join(output_path, new_file_name)
            copy_image = deepcopy(image)
            aug_image = self.augment(copy_image)
            aug_image = cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(new_file_path, aug_image, [cv2.IMWRITE_JPEG_QUALITY, 50])
        logger.info('Complete to output augmented image to destination path.')

    def split_train_test_data(self,
                              source_folder: str,
                              train_folder: str,
                              test_folder: str,
                              train_file: str,
                              test_file: str,
                              test_size=100):
        if not source_folder or not isinstance(source_folder, str) or not os.path.exists(source_folder):
            return
        if not train_folder and not isinstance(train_folder, str):
            return
        if not test_folder and not isinstance(test_folder, str):
            return
        if not train_file and not isinstance(train_file, str):
            return
        if not test_file and not isinstance(test_file, str):
            return

        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
        all_img_files = glob(os.path.join(source_folder, '*.jpg'))
        train_dataset, test_dataset = train_test_split(all_img_files, test_size=test_size, random_state=8)
        origin_dataset = [file for file in test_dataset if 'origin' in file]
        for origin in origin_dataset:
            test_dataset.remove(origin)
        train_dataset += origin_dataset
        logger.info(f'Handle {len(train_dataset)} train image files')
        self.get_label_data(train_dataset, dest_path=train_folder, label_file=train_file)
        logger.info(f'Handle {len(test_dataset)} test image files')
        self.get_label_data(test_dataset, dest_path=test_folder, label_file=test_file)

    def get_label_data(self, image_list: list, dest_path: str, label_file: str):
        os.makedirs(dest_path, exist_ok=True)
        data_list = []
        for index, image_file in enumerate(tqdm(image_list)):
            base_name = os.path.basename(image_file)
            pure_name = base_name.replace('.jpg', '')
            name_list = pure_name.split('_')
            if len(name_list) == 4:
                label_name = '_'.join(name_list[0:3])
                data = {'image': base_name,
                        'label': label_name}
                data_list.append(data)
                shutil.move(image_file, os.path.join(dest_path, base_name))
        label_data = pd.DataFrame(data_list)
        label_data.to_csv(label_file, encoding='utf-8', index=False)
        return data_list




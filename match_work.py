import json
import os
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

from data.data_process import DataProcessor
from keras_cls.utils.preprocess import normalize, resize_img
from keras_cls.train import Image_Classification_Parameter

from keras_match.model.match_net import MatchNet
from keras_match.generator.default_generator import DefaultGenerator
from keras_match.generator.generator_builder import get_generator


def get_image_classification_parameter(data_mode: str = 'train'):
    config = get_config()
    root_path = config.get('root_folder', '/data/math_research/test_paper_cls/')
    try:
        home_path = os.environ['HOME']
        root_path = os.path.join(home_path,
                                 root_path[1:])
    except:
        pass

    dataset_path = config.get('dataset_folder', 'dataset')
    data_folder = config.get(f'{data_mode}_folder', f'{data_mode}')
    data_path = os.path.join(root_path, dataset_path, data_folder)
    data_file = os.path.join(root_path, dataset_path,
                             config.get(f'{data_mode}_match_data_file', f'{data_mode}_match.csv'))
    model_folder = config.get('match_model_folder', 'match_model')

    img_cls_params = Image_Classification_Parameter()
    img_cls_params.checkpoints = os.path.join(root_path, model_folder)
    os.makedirs(img_cls_params.checkpoints, exist_ok=True)
    img_cls_params.label_file = data_file
    img_cls_params.dataset_dir = data_path
    img_cls_params.init_lr = 1e-4
    img_cls_params.optimizer = 'SAM-Adam'
    img_cls_params.num_classes = 2

    # 目前的p3.2xlarge EC2，只能使用EfficienetNetB4 + 4 batch size进行训练，否则会显存溢出
    img_cls_params.backbone = r'EfficientNetB4'
    img_cls_params.progressive_resizing = [(380, 380)]
    img_cls_params.batch_size = 4
    img_cls_params.epochs = 100
    return img_cls_params


def test_match_net():
    image_classification_parameter = get_image_classification_parameter(data_mode='train')
    match_net = MatchNet(image_classification_parameter)

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
    print(pred_result)
    pred_cls = np.argmax(pred_result, axis=-1)
    print(pred_cls)
    pred_result = model([image_list[1], image_list[0]])
    print(pred_result)
    pred_cls = np.argmax(pred_result, axis=-1)
    print(pred_cls)

    print("hello")


def get_config():
    config_file = r'./config/data_config.json'
    with open(config_file, mode='r', encoding='utf-8') as read:
        config = json.load(read)
    return config


def generate_match_data():
    data_process = DataProcessor()
    config = get_config()
    root_path = config.get('root_folder', '/data/math_research/test_paper_cls/')
    dest_path = config.get('dataset_folder', 'dataset')
    train_folder = config.get('train_folder', 'train')
    train_path = os.path.join(root_path, dest_path, train_folder)
    test_folder = config.get('test_folder', 'test')
    test_path = os.path.join(root_path, dest_path, test_folder)

    train_match_data_file = os.path.join(root_path, dest_path, config.get('train_match_data_file', 'train_match.csv'))
    test_match_data_file = os.path.join(root_path, dest_path, config.get('test_match_data_file', 'test_match.csv'))

    data_process.generate_match_data(image_path=train_path, label_file=train_match_data_file)
    data_process.generate_match_data(image_path=test_path, label_file=test_match_data_file)


def test_data_generator():
    image_classification_parameter = get_image_classification_parameter(data_mode='test')
    train_generator, val_generator = get_generator(image_classification_parameter)
    train_generator_tqdm = tqdm(enumerate(train_generator), total=len(train_generator))
    for batch_index, (batch_imgs, batch_labels) in train_generator_tqdm:
        print(len(batch_imgs))


def train_match_model():
    image_classification_parameter = get_image_classification_parameter(data_mode='train')
    match_net = MatchNet(img_cls_params=image_classification_parameter)
    match_net.train_image_classification_model()


if __name__ == '__main__':
    # test_match_net()
    # generate_match_data()
    # test_data_generator()
    train_match_model()

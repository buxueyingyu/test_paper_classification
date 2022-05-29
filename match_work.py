import json
import os
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

from data.data_process import DataProcessor
from keras_cls.utils.preprocess import normalize, resize_img
from keras_cls.train import Image_Tast_Parameter

from keras_match.model.match_net import MatchNet
from keras_match.model.simple_match_net import SimpleMatchNet
from keras_match.model.match_net_inference import MatchNetInference
from keras_match.generator.generator_builder import get_generator


def get_image_task_parameter(data_mode: str = 'train',
                             loss='ce',
                             init_lr=1e-4,
                             backbone: str = 'EfficientNetB4',
                             progressive_resizing=[(256, 256)],
                             num_classes: int = 2,
                             batch_size: int = 8,
                             epochs: int = 20,
                             model_path: str = 'match_model',
                             is_simple_network: bool = False,
                             simple_network_type: str = 'complex_cnn',
                             save_all_epoch_model: bool = False):
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
    img_cls_params = Image_Tast_Parameter()
    img_cls_params.checkpoints = os.path.join(root_path, model_path)
    os.makedirs(img_cls_params.checkpoints, exist_ok=True)
    img_cls_params.label_file = data_file
    img_cls_params.dataset_dir = data_path
    img_cls_params.init_lr = init_lr
    img_cls_params.loss = loss
    img_cls_params.optimizer = 'SAM-Adam'
    img_cls_params.num_classes = num_classes
    img_cls_params.is_simple_network = is_simple_network
    # 目前的p3.2xlarge EC2，只能使用EfficienetNetB4 + 4 batch size进行训练，否则会显存溢出
    img_cls_params.backbone = backbone
    img_cls_params.concat_max_and_average_pool = False
    input_shape = get_input_shape(backbone, progressive_resizing, is_simple_network)
    img_cls_params.progressive_resizing = input_shape
    img_cls_params.batch_size = batch_size
    img_cls_params.epochs = epochs
    img_cls_params.simple_network_type = simple_network_type
    img_cls_params.save_all_epoch_model = save_all_epoch_model
    return img_cls_params


def get_input_shape(backbone: str, raw_shape: list, is_simple_network: bool):
    if is_simple_network:
        return raw_shape
    else:
        backbone_shape_dict = {'b0': 224,
                               'b1': 240,
                               'b2': 260,
                               'b3': 300,
                               'b4': 380,
                               'b5': 456,
                               'b6': 528,
                               'b7': 600}
        efficientnet_type = backbone[-2:].lower()
        shape = backbone_shape_dict.get(efficientnet_type, None)
        if shape:
            input_shape = [(shape, shape)]
        else:
            input_shape = raw_shape
        return input_shape


def test_match_net():
    image_classification_parameter = get_image_task_parameter(data_mode='train')
    match_net = MatchNet(img_cls_params=image_classification_parameter, single_backbone=True)

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
    image_classification_parameter = get_image_task_parameter(data_mode='test')
    train_generator, val_generator = get_generator(image_classification_parameter)
    train_generator_tqdm = tqdm(enumerate(train_generator), total=len(train_generator))
    for batch_index, (batch_imgs, batch_labels) in train_generator_tqdm:
        print(len(batch_imgs))


def train_match_model():
    image_classification_parameter = get_image_task_parameter(data_mode='train')
    match_net = MatchNet(img_cls_params=image_classification_parameter, single_backbone=True)
    match_net.train()


def validate():
    img_cls_param = get_image_task_parameter(data_mode='test')
    match_net_inference = MatchNetInference(img_cls_params=img_cls_param, single_backbone=True)
    match_net_inference.validate(img_cls_param.label_file)


def predict():
    image_path = r'/data/math_research/test_paper_cls/dataset/test/'
    image_pair_list = [['paper_20220506_29_0.jpg', 'paper_20220506_29_10.jpg']]
    img_cls_param = get_image_task_parameter(data_mode='test')
    match_net_inference = MatchNetInference(img_cls_params=img_cls_param, single_backbone=True)
    for index, image_pair in enumerate(tqdm(image_pair_list)):
        match_net_inference.predict(os.path.join(image_path, image_pair[0]),
                                    os.path.join(image_path, image_pair[1]))


def test_simple_match_net():
    image_classification_parameter = get_image_task_parameter(data_mode='train')
    match_net = SimpleMatchNet(img_task_params=image_classification_parameter,
                               is_simple_network=True,
                               single_backbone=True)

    root_path = r'/data/math_research/test_paper_cls/dataset/train/'
    image_file_list = [(r'paper_20220506_2_2.jpg', r'paper_20220506_5_origin.jpg'),
                       (r'paper_20220506_7_8.jpg', r'paper_20220506_47_origin.jpg')]
    img_size = match_net.img_task_params.progressive_resizing[0]
    batch_images = []
    for left_image_file, right_image_file in image_file_list:
        left_image_path = os.path.join(root_path, left_image_file)
        left_image = get_image_data(left_image_path, img_size)

        right_image_path = os.path.join(root_path, right_image_file)
        right_image = get_image_data(right_image_path, img_size)

        batch_images.append((left_image, right_image))
    batch_images = np.array(batch_images)
    model = match_net.get_match_net()
    print(model.summary())
    pred_result = model([batch_images[:, 0], batch_images[:, 1]])
    # pred_result = model.predict_on_batch(image_list)
    # print(pred_result)
    pred_cls = np.argmax(pred_result, axis=-1)
    print(pred_cls)

    print("hello")


def get_image_data(image_file: str, image_size: tuple):
    image = np.ascontiguousarray(Image.open(image_file).convert('RGB'))
    image, _, _ = resize_img(image, image_size)
    image = normalize(image, mode='tf')
    return image


model_dict = {'efficientnet': {'data_mode': 'train',
                               'init_lr': 8e-4,
                               'backbone': 'EfficientB2',
                               'progressive_resizing': [(260, 260)],
                               'num_classes': 2,
                               'batch_size': 16,
                               'epochs': 20,
                               'model_path': 'match_efficientnet_model',
                               'is_simple_network': False,
                               'simple_network_type': None,
                               'save_all_epoch_model': True},
              'pure_dense': {'data_mode': 'train',
                             'init_lr': 8e-4,
                             'backbone': None,
                             'progressive_resizing': [(256, 256)],
                             'num_classes': 2,
                             'batch_size': 8,
                             'epochs': 20,
                             'model_path': 'match_pure_dense_model',
                             'is_simple_network': True,
                             'simple_network_type': 'pure_dense',
                             'save_all_epoch_model': True},
              'complex_cnn': {'data_mode': 'train',
                              'init_lr': 8e-4,
                              'backbone': None,
                              'progressive_resizing': [(256, 256)],
                              'num_classes': 2,
                              'batch_size': 64,
                              'epochs': 20,
                              'model_path': 'match_complex_cnn_model',
                              'is_simple_network': True,
                              'simple_network_type': 'complex_cnn',
                              'save_all_epoch_model': True}
              }


def train_simple_match_model():
    for model_type, model_info in model_dict.items():
        if not model_info:
            return
        logger.info(f'Train model: {model_type}')
        logger.info(model_info)
        img_task_param = get_image_task_parameter(data_mode=model_info.get('data_mode', 'train'),
                                                  init_lr=model_info.get('init_lr', 8e-4),
                                                  backbone=model_info.get('backbone', 'EfficientB2'),
                                                  progressive_resizing=model_info.get('progressive_resizing', [(256, 256)]),
                                                  num_classes=model_info.get('num_classes', 2),
                                                  batch_size=model_info.get('batch_size', 8),
                                                  epochs=model_info.get('epochs', 20),
                                                  model_path=model_info.get('model_path', 'match_model'),
                                                  is_simple_network=model_info.get('is_simple_network', True),
                                                  simple_network_type=model_info.get('simple_network_type', 'complex_cnn'),
                                                  save_all_epoch_model=model_info.get('save_all_epoch_model', False)
                                                  )
        match_net = SimpleMatchNet(img_task_params=img_task_param,
                                   complex_cnn_last_dense_size=256)
        match_net.train()


def validate_simple_match_model():
    model_type = 'efficientnet'
    model_info = model_dict.get(model_type, {})
    if not model_info:
        return
    img_task_param = get_image_task_parameter(data_mode='test',
                                              init_lr=model_info.get('init_lr', 8e-4),
                                              backbone=model_info.get('backbone', 'EfficientB2'),
                                              progressive_resizing=model_info.get('progressive_resizing', [(256, 256)]),
                                              num_classes=model_info.get('num_classes', 2),
                                              batch_size=4,
                                              epochs=model_info.get('epochs', 20),
                                              model_path=model_info.get('model_path', 'match_model'),
                                              is_simple_network=model_info.get('is_simple_network', True),
                                              simple_network_type=model_info.get('simple_network_type', 'complex_cnn'),
                                              save_all_epoch_model=model_info.get('save_all_epoch_model', False)
                                              )
    match_net_inference = MatchNetInference(img_cls_params=img_task_param,
                                            complex_cnn_last_dense_size=256)
    match_net_inference.validate(img_task_param.label_file)


def predict_simple_match_model():
    model_type = 'efficientnet'
    model_info = model_dict.get(model_type, {})
    if not model_info:
        return
    image_path = r'/data/math_research/test_paper_cls/dataset/test/'
    image_pair_list = [['paper_20220506_29_0.jpg', 'paper_20220506_29_10.jpg']]
    img_task_param = get_image_task_parameter(data_mode=model_info.get('data_mode', 'train'),
                                              init_lr=model_info.get('init_lr', 8e-4),
                                              backbone=model_info.get('backbone', 'EfficientB2'),
                                              progressive_resizing=model_info.get('progressive_resizing', [(256, 256)]),
                                              num_classes=model_info.get('num_classes', 2),
                                              batch_size=model_info.get('batch_size', 8),
                                              epochs=model_info.get('epochs', 20),
                                              model_path=model_info.get('model_path', 'match_model'),
                                              is_simple_network=model_info.get('is_simple_network', True),
                                              simple_network_type=model_info.get('simple_network_type', 'complex_cnn'),
                                              save_all_epoch_model=model_info.get('save_all_epoch_model', False)
                                              )
    match_net_inference = MatchNetInference(img_cls_params=img_task_param,
                                            complex_cnn_last_dense_size=256)
    for index, image_pair in enumerate(tqdm(image_pair_list)):
        print(match_net_inference.predict(os.path.join(image_path, image_pair[0]),
                                          os.path.join(image_path, image_pair[1])))


def batch_predict_simple_match_model():
    model_type = 'efficientnet'
    model_info = model_dict.get(model_type, {})
    if not model_info:
        return
    img_task_param = get_image_task_parameter(data_mode='test',
                                              init_lr=model_info.get('init_lr', 8e-4),
                                              backbone=model_info.get('backbone', 'EfficientB2'),
                                              progressive_resizing=model_info.get('progressive_resizing', [(256, 256)]),
                                              num_classes=model_info.get('num_classes', 2),
                                              batch_size=4,
                                              epochs=model_info.get('epochs', 20),
                                              model_path=model_info.get('model_path', 'match_model'),
                                              is_simple_network=model_info.get('is_simple_network', True),
                                              simple_network_type=model_info.get('simple_network_type', 'complex_cnn'),
                                              save_all_epoch_model=model_info.get('save_all_epoch_model', False)
                                              )
    match_net_inference = MatchNetInference(img_cls_params=img_task_param,
                                            complex_cnn_last_dense_size=256)
    match_net_inference.batch_predict()


if __name__ == '__main__':
    # test_match_net()
    # generate_match_data()
    # test_data_generator()
    # train_match_model()
    # predict()
    # validate()
    # predict_simple_match_model()
    # test_simple_match_net()
    # train_simple_match_model()
    validate_simple_match_model()
    # batch_predict_simple_match_model()
import json
import os
import cv2
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

from data.data_process import DataProcessor
from keras_cls.train import \
    train_image_classification_model, Image_Classification_Parameter
from keras_cls.inference import get_image_classification_inference


def get_config():
    config_file = r'./config/data_config.json'
    with open(config_file, mode='r', encoding='utf-8') as read:
        config = json.load(read)
    return config


def rename_raw_file():
    config = get_config()
    data_process = DataProcessor()
    root_path = config.get('root_folder', '/data/math_research/test_paper_cls/')
    origin_image_folder = config.get('origin_image_folder', 'raw_data/origin_image')

    data_process.rename_files(os.path.join(root_path, origin_image_folder))


def augment_image():
    image_file = r'/data/math_research/test_paper_cls/raw_data/origin_image/paper_20220506_22.jpg'
    data_process = DataProcessor()
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i in range(9):
        aug_image = data_process.light_aug(image)
        plt.subplot(3, 3, i + 1)
        plt.imshow(aug_image)
    plt.show()


def output_augment_files():
    config = get_config()
    root_path = config.get('root_folder', '/data/math_research/test_paper_cls/')
    origin_image_folder = config.get('origin_image_folder', 'raw_data/origin_image')
    origin_path = os.path.join(root_path, origin_image_folder)
    origin_image_files = glob(os.path.join(origin_path, '*.jpg'))
    augment_image_folder = config.get('augment_image_folder', 'raw_data/augment_image')
    output_path = os.path.join(root_path, augment_image_folder)
    os.makedirs(output_path, exist_ok=True)


def compress_image():
    """
    经验证，cv2.IMWRITE_JPEG_QUALITY 设置为50刚好
    需要注意的是，保存图片为文件前，需要先执行
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    否则颜色会不正确
    实证，压缩后的图片，对segmentation模型的预测结果影响不大
    Returns:

    """
    # raw_image_file = r'C:\data\math_research\test_paper_cls\raw_data\augment_image\paper_20220506_28_2.jpg'
    # image = cv2.imread(raw_image_file)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(r'C:\data\math_research\test_paper_cls\raw_data\augment_image\paper_20220506_28_2_copy.jpg',
    #             image,
    #             [cv2.IMWRITE_JPEG_QUALITY, 50])
    raw_image_path = r'/data/val/fixed_label_img/'
    dest_image_path = r'/data/fixed_label_img_compressed/'
    raw_image_files = glob(os.path.join(raw_image_path, '*.jpg'))
    for index, raw_image_file in enumerate(tqdm(raw_image_files)):
        image = cv2.imread(raw_image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(dest_image_path, os.path.basename(raw_image_file)),
                    image,
                    [cv2.IMWRITE_JPEG_QUALITY, 50])
        if index == 10:
            break


def compress_image_by_byte():
    image_file = r'C:\data\math_research\test_paper_cls\raw_data\augment_image\paper_20220506_10_origin.jpg'
    image = cv2.imread(image_file)

    image_copy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(r'C:\data\math_research\test_paper_cls\raw_data\paper_20220506_10_origin_1.jpg',
                image_copy,
                [cv2.IMWRITE_JPEG_QUALITY, 50])

    # 取值范围：0~100，数值越小，压缩比越高，图片质量损失越严重
    params = [cv2.IMWRITE_JPEG_QUALITY, 50]  # ratio:0~100
    compressed = cv2.imencode(".jpg", image, params)[1]
    compressed = (np.array(compressed)).tobytes()
    compressed_image = cv2.imdecode(np.frombuffer(compressed, np.uint8), cv2.COLOR_BGR2RGB)
    # data_process = DataProcessor()
    # aug_image = data_process.light_aug(compressed_image)
    compressed_image = cv2.cvtColor(compressed_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(r'C:\data\math_research\test_paper_cls\raw_data\paper_20220506_10_origin.jpg',
                compressed_image)
    # msg = (np.array(compressed_image)).tobytes()
    # print("msg:", len(msg))


def split_train_test():
    config = get_config()
    root_path = config.get('root_folder', '/data/math_research/test_paper_cls/')
    source_image_folder = config.get('augment_image_folder', 'raw_data/augment_image')
    source_path = os.path.join(root_path, source_image_folder)

    dest_path = config.get('dataset_folder', 'dataset')
    train_folder = config.get('train_folder', 'train')
    train_path = os.path.join(root_path, dest_path, train_folder)
    test_folder = config.get('test_folder', 'test')
    test_path = os.path.join(root_path, dest_path, test_folder)

    train_data_file = os.path.join(root_path, dest_path, config.get('train_data_file', 'train.csv'))
    test_data_file = os.path.join(root_path, dest_path, config.get('test_data_file', 'test.csv'))

    data_process = DataProcessor()
    data_process.split_train_test_data(source_folder=source_path,
                                       train_folder=train_path,
                                       test_folder=test_path,
                                       train_file=train_data_file,
                                       test_file=test_data_file,
                                       test_size=100)


def train():
    config = get_config()
    root_path = config.get('root_folder', '/data/math_research/test_paper_cls/')
    try:
        home_path = os.environ['HOME']
        root_path = os.path.join(home_path,
                                 root_path[1:])
    except:
        pass
    dataset_path = config.get('dataset_folder', 'dataset')
    train_folder = config.get('train_folder', 'train')
    train_path = os.path.join(root_path, dataset_path, train_folder)
    train_data_file = os.path.join(root_path, dataset_path, config.get('train_data_file', 'train.csv'))
    model_folder = config.get('model_folder', 'model')

    img_cls_params = Image_Classification_Parameter()
    img_cls_params.checkpoints = os.path.join(root_path, model_folder)
    os.makedirs(img_cls_params.checkpoints, exist_ok=True)
    img_cls_params.label_file = train_data_file
    img_cls_params.dataset_dir = train_path
    img_cls_params.backbone = r'EfficientNetB5'
    img_cls_params.progressive_resizing = [(456, 456)]
    img_cls_params.epochs = 100
    img_cls_params.batch_size = 1
    train_image_classification_model(img_cls_params)


def validate_image_classification():
    config = get_config()
    root_path = config.get('root_folder', '/data/math_research/test_paper_cls/')
    try:
        home_path = os.environ['HOME']
        root_path = os.path.join(home_path,
                                 root_path[1:])
    except:
        pass
    inference = get_image_classification_inference(root_path=root_path,
                                                   backbone='EfficientNetB5',
                                                   image_size=(456, 456),
                                                   model_folder='model/',
                                                   test_label_file = r'dataset/test.csv',
                                                   test_image_folder = r'dataset/test/')
    bythreshold = True
    threshold_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    for threshold in threshold_list:
        inference.validate(inference.para.label_file, by_threshold=bythreshold, threshold=threshold)

    # inference = get_image_classification_inference(model_folder=r'model/img_cls/backup/')
    # inference.validate(inference.para.label_file)


def classify_table_image(inference=None,
                         dataset_path=r'pdf_table/split/dataset_20210916',
                         img_folder: str = 'img/'):
    if inference is None:
        inference = get_image_classification_inference()

    try:
        home_path = os.environ['HOME']
        root_path = os.path.join(home_path,
                                 r'data/efs/mid/pubtabnet/train/')
    except:
        root_path = r'/data/efs/mid/pubtabnet/train/'

    inference.para.label_file = os.path.join(root_path, dataset_path, r'img_cls.csv')
    inference.para.dataset_dir = os.path.join(root_path, dataset_path, img_folder)

    data_list = []
    img_cls_label_path = os.path.join(root_path, dataset_path, r'img_cls_label/')
    if not os.path.exists(img_cls_label_path):
        os.makedirs(img_cls_label_path)

    img_list = glob(os.path.join(inference.para.dataset_dir, '*.jpg'))
    img_amount = len(img_list)
    if img_list:
        for index, img_path in enumerate(tqdm(img_list)):
            prediction_value = inference.predict(img_path, by_threshold=True, threshold=0.5)
            prediction_name = inference.idx2tag.get(prediction_value, 'normal')
            logger.info('Handle the {0}/{1} image: {2}, class: {3}'
                        .format(index + 1,
                                img_amount,
                                os.path.basename(img_path),
                                prediction_name))
            data = {'image': os.path.basename(img_path), 'label': prediction_name}
            data_list.append(data)
    #         write_label_xml_file(img_cls_label_path, img_path, prediction_name)
    # data_df = pd.DataFrame(data_list)
    # data_df.to_csv(inference.para.label_file, sep=',', encoding='utf-8', index=False)


if __name__ == '__main__':
    # rename_raw_file()
    # augment_image()
    # output_augment_files()
    # compress_image()
    # compress_image_by_byte()
    # split_train_test()
    train()

from data.data_process import DataProcessor
import json
import os
import cv2
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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

    data_process = DataProcessor()
    for index, origin_image_file in enumerate(tqdm(origin_image_files)):
        data_process.output_augment_image_file(image_file=origin_image_file,
                                               output_path=output_path,
                                               augment_amounts=20)


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


if __name__ == '__main__':
    # rename_raw_file()
    # augment_image()
    # output_augment_files()
    compress_image()
    # compress_image_by_byte()
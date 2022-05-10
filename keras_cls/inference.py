import os, argparse, sys
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from sklearn import metrics as sklearn_metrics
import time
from tqdm import tqdm

from keras_cls.model.model_builder import get_model
from keras_cls.train import Image_Classification_Parameter
from keras_cls.utils.common import get_best_model_path
from keras_cls.utils.preprocess import normalize, resize_img


class Image_Classification_Inference:
    def __init__(self, para: Image_Classification_Parameter):
        self.para = para
        checkpoints = para.checkpoints
        self.tag2idx = None
        if checkpoints and isinstance(checkpoints, str) and os.path.exists(checkpoints):
            class_name_file = os.path.join(checkpoints, 'class.names')
            if os.path.exists(class_name_file):
                with open(class_name_file, encoding='utf-8', mode='r') as c_file:
                    lines = c_file.readlines()
                    self.tag2idx = {line.strip(): index
                                    for index, line
                                    in enumerate(lines)
                                    if len(line.strip()) > 0}
                    self.idx2tag = {index: line.strip()
                                    for index, line
                                    in enumerate(lines)
                                    if len(line.strip()) > 0}
        if para.num_classes:
            self.model = get_model(para, para.num_classes)
        elif self.tag2idx:
            self.model = get_model(para, len(self.tag2idx.keys()))
        else:
            self.model = get_model(para, 3)
        local_weights = para.local_weights
        if local_weights and isinstance(local_weights, str) and os.path.exists(local_weights):
            print(f'local weights: {local_weights}')
            self.model.load_weights(local_weights)
        else:
            checkpoints = para.checkpoints
            if checkpoints and isinstance(checkpoints, str) and os.path.exists(checkpoints):
                local_weights = get_best_model_path(checkpoints)
                self.model.load_weights(local_weights)
                print(f'table classification checkpoints: {checkpoints}, local weights: {local_weights}')
        self.local_weights = local_weights
        self.baseline = Inference_Baseline()

    def validate(self, validate_data_file: str):
        assert validate_data_file and os.path.exists(validate_data_file)
        validate_data = pd.read_csv(validate_data_file, sep=',')

        if self.tag2idx is None:
            label_list = sorted(list(set(list(validate_data['label']))))
            self.tag2idx = {label.strip(): index
                            for index, label
                            in enumerate(label_list)
                            if len(label.strip()) > 0}
            self.idx2tag = {index: label.strip()
                            for index, label
                            in enumerate(label_list)
                            if len(label.strip()) > 0}
        true_list = []
        predict_list = []
        tags = list(self.tag2idx.keys())
        details = []
        for index, row in tqdm(validate_data.iterrows()):
            label = row['label']
            image_name = row['image']
            image_path = os.path.join(self.para.dataset_dir, image_name)
            if os.path.exists(image_path):
                ground_truth = self.tag2idx.get(label, 1)
                prediction_value = self.predict(image_path)
                print('{0}/{1} Image: {2}, ground truth: {3}, prediction: {4}'
                      .format(index + 1,
                              len(validate_data),
                              image_name,
                              self.idx2tag.get(ground_truth, 'N/A'),
                              self.idx2tag.get(prediction_value, 'N/A')))

                true_list.append(ground_truth)
                predict_list.append(prediction_value)
                data = {'image': image_name,
                        'ground_truth': self.idx2tag.get(ground_truth, 'N/A'),
                        'prediction': self.idx2tag.get(prediction_value, 'N/A')}
                details.append(data)
        labels = [label for label in range(len(tags))]
        report = sklearn_metrics.classification_report(true_list,
                                                       predict_list,
                                                       labels=labels,
                                                       target_names=tags)

        print(report)
        if self.para.checkpoints:
            if self.local_weights:
                local_weights_basename = os.path.basename(self.local_weights)
            else:
                local_weights_basename = r'img_cls_best_weight'
            report_folder = os.path.join(self.para.checkpoints, 'report/')
            os.makedirs(report_folder, exist_ok=True)
            time_text = time.strftime('%Y%m%d%H%M%S',
                                      time.localtime(
                                          time.time()))
            report_file = f'{local_weights_basename}_{time_text}.txt'
            report_path = os.path.join(report_folder, report_file)
            with open(report_path, mode='w', encoding='utf-8') as file:
                file.write(report)
            detail_df = pd.DataFrame(details)

            detail_file = f'{local_weights_basename}_{time_text}.csv'
            detail_path = os.path.join(report_folder, detail_file)
            detail_df.to_csv(detail_path, sep=',', encoding='utf-8', index=False)

    def predict(self, image_path: str):
        """
        :param image_path:
        :type image_path:
        :return:
        :rtype:
        """
        image = np.ascontiguousarray(Image.open(image_path).convert('RGB'))
        image, _, _ = resize_img(image, self.para.progressive_resizing[0])
        image = self.baseline.distort(image)
        # self.show_img(image_path, image)
        image = normalize(image, mode='tf')
        pred_result = self.model.predict_on_batch(image)
        pred_cls = np.argmax(pred_result, axis=-1)
        return pred_cls[0]

    def predict_cls_name(self, image_path: str):
        pred_cls = self.predict(image_path)
        return self.idx2tag.get(pred_cls, 'N/A')

    def show_img(self, image_path, image):
        plt.figure(os.path.basename(image_path))
        plt.imshow(image)
        plt.axis('on')
        plt.title(os.path.basename(image_path))
        plt.show()


class Inference_Baseline:
    def __init__(self, ):
        self.transform = A.Compose([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1.0),
        ])

    def distort(self, img):
        img = self.transform(image=img)['image']
        return img


def get_image_classification_inference(root_path: str,
                                       backbone: str = 'EfficientNetB5',
                                       image_size=(456, 456),
                                       model_folder: str = r'model/',
                                       test_label_file: str = r'dataset/test.csv',
                                       test_image_folder: str = r'dataset/test/'):
    try:
        home_path = os.environ['HOME']
        root_path = os.path.join(home_path, root_path[1:])
    except:
        pass

    img_cls_params = Image_Classification_Parameter()
    img_cls_params.checkpoints = os.path.join(root_path, model_folder)
    img_cls_params.label_file = os.path.join(root_path,
                                             test_label_file)
    img_cls_params.dataset_dir = os.path.join(root_path, test_image_folder)

    img_cls_params.backbone = backbone
    img_cls_params.progressive_resizing = [image_size]
    img_cls_params.epochs = 100
    img_cls_params.batch_size = 8

    try:
        inference = Image_Classification_Inference(img_cls_params)
    except Exception as e:
        inference = None
        print(f'Load table classification error: {e}')
    return inference


if __name__ == '__main__':
    pass

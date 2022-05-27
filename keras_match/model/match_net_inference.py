import os, argparse, sys
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from sklearn import metrics as sklearn_metrics
import time
from tqdm import tqdm

from keras_cls.train import Image_Tast_Parameter
from keras_cls.utils.common import get_best_model_path
from keras_cls.utils.preprocess import normalize, resize_img
from keras_match.model.match_net import MatchNet
from keras_match.model.simple_match_net import SimpleMatchNet


class MatchNetInference:
    def __init__(self,
                 img_cls_params: Image_Tast_Parameter,
                 complex_cnn_last_dense_size=256):
        self.para = img_cls_params
        checkpoints = img_cls_params.checkpoints
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
        self.is_simple_net = img_cls_params.is_simple_network
        match_net = SimpleMatchNet(img_task_params=img_cls_params,
                                   complex_cnn_last_dense_size=complex_cnn_last_dense_size)
        self.model = match_net.get_match_net()
        try:
            local_weights = img_cls_params.local_weights
            if local_weights and isinstance(local_weights, str) and os.path.exists(local_weights):
                print(f'local weights: {local_weights}')
                self.model.load_weights(local_weights)
            else:
                checkpoints = img_cls_params.checkpoints
                if checkpoints and isinstance(checkpoints, str) and os.path.exists(checkpoints):
                    local_weights = get_best_model_path(checkpoints, prefix=r'img_match*')
                    if local_weights:
                        self.model.load_weights(local_weights)
                        print(f'table classification checkpoints: {checkpoints}, local weights: {local_weights}')
            self.local_weights = local_weights
        except:
            pass
        img_size = img_cls_params.progressive_resizing[-1]
        self.baseline = Inference_Baseline(img_size)

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
            left_image_name = row['left']
            left_image_path = os.path.join(self.para.dataset_dir, left_image_name)
            right_image_name = row['right']
            right_image_path = os.path.join(self.para.dataset_dir, right_image_name)
            if os.path.exists(left_image_path) and os.path.exists(right_image_path):
                ground_truth = self.tag2idx.get(str(label), 1)
                prediction_cls, prediction_value = self.predict(left_image_path,
                                                                right_image_path,
                                                                with_pred_value=True)
                if ground_truth == 1:
                    print(
                        '{0}/{1} left mage: {2}, right image: {3}, ground truth: {4}, prediction: {5}, prediction value: {6}'
                        .format(index + 1,
                                len(validate_data),
                                left_image_name,
                                right_image_name,
                                self.idx2tag.get(ground_truth, 'N/A'),
                                self.idx2tag.get(prediction_cls, 'N/A'),
                                prediction_value))

                true_list.append(ground_truth)
                predict_list.append(prediction_cls)
                data = {'left_image': left_image_name,
                        'right_image': right_image_name,
                        'ground_truth': self.idx2tag.get(ground_truth, 'N/A'),
                        'prediction': self.idx2tag.get(prediction_cls, 'N/A'),
                        'prediction_value': prediction_value}
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
                local_weights_basename = r'match_net_best_weight'
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

    def batch_predict(self):
        from keras_match.generator.generator_builder import get_generator
        import tensorflow as tf
        self.para.batch_size=4
        train_generator, val_generator = get_generator(self.para)
        val_generator_tqdm = tqdm(enumerate(val_generator), total=len(val_generator))
        val_acc = 0
        num_img = 0
        for batch_index, (batch_imgs, batch_labels) in val_generator_tqdm:
            model_outputs = self.model([batch_imgs[:, 0], batch_imgs[:, 1]])
            raw_result = np.array(tf.less(model_outputs, 0.5))
            predict_result = [[int(result[0])] for result in raw_result]
            wrong_pred_mask = (batch_labels == predict_result)
            val_acc += np.sum(wrong_pred_mask)
            num_img += np.shape(batch_labels)[0]
            cur_val_acc = val_acc / num_img
            val_generator_tqdm.set_description(
                "val_acc:{:.5f}".format(cur_val_acc))

    def predict(self,
                left_image_path: str,
                right_image_path: str,
                with_pred_value: bool = False):
        """

        Args:
            left_image_path ():
            right_image_path ():

        Returns:

        """
        left_image = self.get_image_data(image_path=left_image_path)
        right_image = self.get_image_data(image_path=right_image_path)
        batch_images = np.array([(left_image, right_image)])
        pred_result = self.model([batch_images[:, 0], batch_images[:, 1]])
        pred_cls = int(pred_result[0][0] < 0.5)
        if with_pred_value:
            return pred_cls, float(pred_result[0][0])
        else:
            return pred_cls

    def get_image_data(self, image_path):
        image = np.ascontiguousarray(Image.open(image_path).convert('RGB'))
        image, _, _ = resize_img(image, self.para.progressive_resizing[0])
        image = self.baseline.distort(image)
        image = normalize(image, mode=None)
        # image = image.reshape(image.shape[0], image.shape[1], 1)
        return image

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
    def __init__(self, img_size: tuple):
        self.transform = A.Compose([
            A.CenterCrop(width=img_size[0], height=img_size[1]),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1.0),
        ])

    def distort(self, img):
        img = self.transform(image=img)['image']
        return img


if __name__ == '__main__':
    pass

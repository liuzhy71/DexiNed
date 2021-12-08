import os
import random

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from scipy import ndimage as ndi
from skimage.morphology import dilation, erosion
from skimage.segmentation import find_boundaries
from scipy import ndimage as ndi

resize_scale = 1

DATASET_NAMES = [
    'BIPED',
    'BIPEDv2',
    'BSDS',
    'BSDS2',
    'BSDS300',
    'CID',
    'DCD',
    'MDBD',  # 5
    'PASCAL',
    'NYUD',
    'CLASSIC',
    'cabin',
    'ML-Hypersim'
]  # 12


def dataset_info(dataset_name, is_linux=True):
    if is_linux:

        config = {
            'BSDS': {
                'img_height': 512,  # 321
                'img_width': 512,  # 481
                'train_list': 'train_pair.lst',
                'test_list': 'test_pair.lst',
                'data_dir': '/opt/dataset/BSDS',  # mean_rgb
                'yita': 0.5
            },
            'BSDS2': {
                'img_height': 512,  # 321
                'img_width': 512,  # 481
                'train_list': 'train_pair2.lst',
                'test_list': 'test_pair.lst',
                'data_dir': '/opt/dataset/BSDS',  # mean_rgb
                'yita': 0.5
            },
            'BSDS300': {
                'img_height': 512,  # 321
                'img_width': 512,  # 481
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/opt/dataset/BSDS300',  # NIR
                'yita': 0.5
            },
            'PASCAL': {
                'img_height': 400,  # 375
                'img_width': 512,  # 500
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/opt/dataset/PASCAL',  # mean_rgb
                'yita': 0.3
            },
            'CID': {
                'img_height': 512,
                'img_width': 512,
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/opt/dataset/CID',  # mean_rgb
                'yita': 0.3
            },
            'NYUD': {
                'img_height': 448,  # 425
                'img_width': 560,  # 560
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/opt/dataset/NYUD',  # mean_rgb
                'yita': 0.5
            },
            'MDBD': {
                'img_height': 720,
                'img_width': 1280,
                'test_list': 'test_pair.lst',
                'train_list': 'train_pair.lst',
                'data_dir': '/opt/dataset/MDBD',  # mean_rgb
                'yita': 0.3
            },
            'BIPED': {
                'img_height': 720,  # 720 # 1088
                'img_width': 1280,  # 1280 5 1920
                'test_list': 'test_rgb.lst',
                'train_list': 'train_rgb.lst',
                'data_dir': 'data/archive/BIPED/edges',  # mean_rgb
                'yita': 0.5
            },
            'BIPEDv2': {
                'img_height': 720,  # 720 # 1088
                'img_width': 1280,  # 1280 5 1920
                'test_list': 'test_pair.lst',
                'train_list': 'train_pair.lst',
                'data_dir': 'data/archive/BIPEDv2/edges/',  # mean_rgb
                'yita': 0.5
            },
            'CLASSIC': {
                'img_height': 768,
                'img_width': 1024,
                'test_list': None,
                'train_list': None,
                'data_dir': 'data/archive/Classic',  # mean_rgb
                'yita': 0.5
            },
            'cabin': {
                'img_height': 464,
                'img_width': 512,
                'test_list': None,
                'train_list': None,
                'data_dir': 'data/archive/cabin',  # mean_rgb
                'yita': 0.5
            },
            'DCD': {
                'img_height': 240,  # 240
                'img_width': 368,  # 360
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/opt/dataset/DCD',  # mean_rgb
                'yita': 0.2
            },
            'ML-Hypersim': {
                'img_height': 768,  # 240
                'img_width': 1024,  # 360
                'test_list': './data/archive/ml-hypersim/hypersim_test.txt',
                'train_list': './data/archive/ml-hypersim/hypersim_train.txt',
                'data_dir': '/home/ubuntu/data/Downloads/ml-hypersim/ml-hypersim/evermotion_dataset/scenes',  # mean_rgb
                'yita': 0.2}
        }
    else:
        config = {
            'BSDS': {'img_height': 720,  # 321
                     'img_width': 720,  # 481
                     'test_list': 'test_pair.lst',
                     'data_dir': '../../dataset/BSDS',  # mean_rgb
                     'yita': 0.5},
            'BSDS300': {'img_height': 512,  # 321
                        'img_width': 512,  # 481
                        'test_list': 'test_pair.lst',
                        'data_dir': '../../dataset/BSDS300',  # NIR
                        'yita': 0.5},
            'PASCAL': {'img_height': 375,
                       'img_width': 500,
                       'test_list': 'test_pair.lst',
                       'data_dir': '/opt/dataset/PASCAL',  # mean_rgb
                       'yita': 0.3},
            'CID': {'img_height': 512,
                    'img_width': 512,
                    'test_list': 'test_pair.lst',
                    'data_dir': '../../dataset/CID',  # mean_rgb
                    'yita': 0.3},
            'NYUD': {'img_height': 425,
                     'img_width': 560,
                     'test_list': 'test_pair.lst',
                     'data_dir': '/opt/dataset/NYUD',  # mean_rgb
                     'yita': 0.5},
            'MDBD': {'img_height': 720,
                     'img_width': 1280,
                     'test_list': 'test_pair.lst',
                     'train_list': 'train_pair.lst',
                     'data_dir': '../../dataset/MDBD',  # mean_rgb
                     'yita': 0.3},
            'BIPED': {'img_height': 720,  # 720
                      'img_width': 1280,  # 1280
                      'test_list': 'test_rgb.lst',
                      'train_list': 'train_rgb.lst',
                      'data_dir': '../../dataset/BIPED/edges',  # WIN: '../.../dataset/BIPED/edges'
                      'yita': 0.5},
            'CLASSIC': {'img_height': 512,
                        'img_width': 512,
                        'test_list': None,
                        'train_list': None,
                        'data_dir': 'data',  # mean_rgb
                        'yita': 0.5},
            'DCD': {'img_height': 240,
                    'img_width': 360,
                    'test_list': 'test_pair.lst',
                    'data_dir': '/opt/dataset/DCD',  # mean_rgb
                    'yita': 0.2},
            'ML-Hypersim': {
                'img_height': 768,  # 240
                'img_width': 1024,  # 360
                'test_list': './data/archive/ml-hypersim/hypersim_test.txt',
                'train_list': './data/archive/ml-hypersim/hypersim_train.txt',
                'data_dir': '/home/ubuntu/data/Downloads/ml-hypersim/ml-hypersim/evermotion_dataset/scenes',  # mean_rgb
                'yita': 0.2}
        }
    return config[dataset_name]


def find_boundaries_color(label_img: np.ndarray, connectivity=1, mode='inner'):
    """Return rgb array where boundaries between labeled regions are colored in
    their corresponding color of the masks.

    Parameters
    ----------
    label_img : array of int or bool
        An array in which different regions are labeled with either different
        integers or boolean values.
    connectivity : int in {1, ..., `label_img.ndim`}, optional
        A pixel is considered a boundary pixel if any of its neighbors
        has a different label. `connectivity` controls which pixels are
        considered neighbors. A connectivity of 1 (default) means
        pixels sharing an edge (in 2D) or a face (in 3D) will be
        considered neighbors. A connectivity of `label_img.ndim` means
        pixels sharing a corner will be considered neighbors.
    mode : only 'inner' mode is supported, probably 'subpixel' could be used
        How to mark the boundaries:
        - inner: outline the pixels *just inside* of objects, leaving
          background pixels untouched.

        - subpixel: return a doubled image, with pixels *between* the
          original pixels marked as boundary where appropriate.

    Returns
    -------
    boundaries : array of rgb values, same shape as `label_img`
        A rgb image where a boundary pixel is in its color of the original mask.
        For `mode` equal to 'subpixel', ``boundaries.shape[i]`` is equal
        to ``2 * label_img.shape[i] - 1`` for all ``i`` (a pixel is
        inserted in between all other pairs of pixels).

    """

    ndim = label_img.ndim
    if ndim == 3:
        if label_img.shape[2] == 4:
            label_img = label_img[:, :, 0:3]
    if ndim == 3:
        selem = ndi.generate_binary_structure(ndim - 1, connectivity)  # ?
    if ndim == 2:
        selem = ndi.generate_binary_structure(ndim, connectivity)  # ?
    else:
        selem = ndi.generate_binary_structure(ndim, connectivity)  # ?
    if mode == 'inner':
        if ndim == 2:
            boundaries = np.zeros(label_img.shape[0:2], dtype=bool)
            boundaries |= dilation(label_img, selem) != erosion(label_img, selem)
            boundaries_out = boundaries.astype(np.uint8) * label_img
        else:
            boundaries = np.zeros(label_img.shape[0:2], dtype=bool)
            for i in range(label_img.shape[2]):
                boundaries |= dilation(label_img[:, :, i], selem) != erosion(label_img[:, :, i], selem)
            boundaries_out = np.expand_dims(boundaries.astype(np.uint8), axis=-1).repeat(3, axis=-1) * label_img
        # img = Image.fromarray(boundaries_color)
        # img.show()
        return boundaries_out
    else:
        raise NotImplementedError


class TestDataset(Dataset):
    def __init__(self,
                 data_root,
                 test_data,
                 mean_bgr,
                 img_height,
                 img_width,
                 test_list=None,
                 arg=None
                 ):
        if test_data not in DATASET_NAMES:
            raise ValueError(f"Unsupported dataset: {test_data}")

        self.data_root = data_root
        self.test_data = test_data
        self.test_list = test_list
        self.args = arg
        # self.arg = arg
        # self.mean_bgr = arg.mean_pixel_values[0:3] if len(arg.mean_pixel_values) == 4 \
        #     else arg.mean_pixel_values
        self.mean_bgr = mean_bgr
        self.img_height = img_height
        self.img_width = img_width
        self.two_type = arg.two_type
        self.data_index = self._build_index()

        print(f"mean_bgr: {self.mean_bgr}")

    def _build_index(self):
        sample_indices = []
        dataset_inf = dataset_info(self.args.test_data)
        if self.test_data == "CLASSIC":
            # for single image testing
            images_path = os.listdir(self.data_root)
            labels_path = None
            sample_indices = [images_path, labels_path]
        elif self.test_data == "cabin":
            # for single image testing
            images_path = os.listdir(self.data_root)
            labels_path = None
            sample_indices = [images_path, labels_path]
        elif self.test_data.lower() == 'biped' or self.test_data.lower() == 'bipedv2':
            images_path = os.path.join(self.data_root,
                                       'imgs',
                                       'test',
                                       'rgbr')
            labels_path = os.path.join(self.data_root,
                                       'edge_maps',
                                       'test',
                                       'rgbr')

            for file_name_ext in os.listdir(images_path):
                file_name = os.path.splitext(file_name_ext)[0]
                sample_indices.append(
                    (os.path.join(images_path, file_name + '.jpg'),
                     os.path.join(labels_path, file_name + '.png'))
                )
        elif self.test_data == 'ML-Hypersim':
            with open(dataset_inf['test_list'], 'r') as f:
                while True:
                    line = f.readline()  # 整行读取数据
                    if not line:
                        break
                    line_sep = line.split(',')
                    scene_path = os.path.join(self.data_root, line_sep[0], 'images')
                    img_id = '%04d' % int(line_sep[2])
                    sample_indices.append(
                        (os.path.join(scene_path, f'scene_{line_sep[1]}_final_preview',
                                      f'frame.{img_id}.tonemap.jpg'),
                         os.path.join(scene_path, f'scene_{line_sep[1]}_geometry_preview',
                                      f'frame.{img_id}.mesh_edge.png'),
                         os.path.join(scene_path, f'scene_{line_sep[1]}_geometry_hdf5',
                                      f'frame.{img_id}.render_entity_id.hdf5'),
                         line_sep[0], line_sep[1], img_id)
                    )
        else:
            # image and label paths are located in a list file

            if not self.test_list:
                raise ValueError(f"Test list not provided for dataset: {self.test_data}")

            list_name = os.path.join(self.data_root, self.test_list)
            with open(list_name, 'r') as f:
                files = f.readlines()
            files = [line.strip() for line in files]
            pairs = [line.split() for line in files]
            images_path = [line[0] for line in pairs]
            labels_path = [line[1] for line in pairs]
            sample_indices = [images_path, labels_path]
        return sample_indices

    def __len__(self):
        return len(self.data_index[0])

    def __getitem__(self, idx):
        # load data
        if self.test_data == 'ML-Hypersim':
            image_path, label_path_geo, label_path_render, scene_name, cam_name, img_id = self.data_index[idx]
            # img_name = os.path.basename(image_path)
            # filetype = os.path.splitext(img_name)[-1]
            # file_name = os.path.splitext(img_name)[0] + ".jpg"

            crop_size_h = int(768 / resize_scale)
            crop_size_w = int(1024 / resize_scale)
            image_path, label_path_geo, label_path_render, scene_name, cam_name, img_id = self.data_index[idx]
            image_cv2 = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_cv2 = cv2.resize(image_cv2, dsize=(crop_size_w, crop_size_h))
            image = np.array(image_cv2).astype(np.float32)

            label_geo_cv2 = cv2.imread(label_path_geo, cv2.IMREAD_COLOR)
            label_geo_cv2 = cv2.resize(label_geo_cv2, dsize=(crop_size_w, crop_size_h))
            label_geo = np.array(label_geo_cv2)  # in color BGR(not RGB)

            condition_blank = (label_geo[:, :, 0] == 255) & (label_geo[:, :, 1] == 255) & (label_geo[:, :, 2] == 255)
            blank = np.where(condition_blank)
            label_geo[blank] = np.array([0, 0, 0])

            if self.two_type:
                condition_SE = (label_geo[:, :, 0] >= 180)  # B channel: PE
                PE = np.where(condition_SE)
                condition_SiE = (label_geo[:, :, 1] >= 180)  # G channel: SiE
                SiE = np.where(condition_SiE)
                condition_PE = (label_geo[:, :, 2] >= 180)  # R channel: SE
                SE = np.where(condition_PE)
                label_geo_code = np.zeros([label_geo.shape[0], label_geo.shape[1]], dtype=int)
                label_geo_code[SE] = 1
                label_geo_code[SiE] = 1
                label_geo_code[PE] = 2
                label_geo = label_geo_code

                label = label_geo[:, :, np.newaxis]

                # with h5py.File(label_path_render, 'r') as f:
                #     label_render = f["dataset"][:].astype(np.int32)
                #     label_render = find_boundaries_color(label_render, mode='inner')
                #     label_render = label_render[:, :, np.newaxis]
                #
                # label = np.concatenate([label_geo, label_render], axis=2).astype(np.float32)
            else:
                # label = cv2.cvtColor(label_geo, cv2.COLOR_BGR2GRAY).astype(float)
                # label = cv2.imread(label_path_geo, cv2.IMREAD_GRAYSCALE).astype(float)

                label = label_geo.sum(axis=2)
                label = np.clip(label, 0, 1)

                # label[np.where(label >= 50)] = 255
                label = label[:, :, np.newaxis]

            im_shape = [image.shape[0], image.shape[1]]
            image = image.transpose([2, 0, 1])
            label = label.transpose([2, 0, 1]).astype(int)
            image = torch.from_numpy(image.copy()).float()
            label = torch.from_numpy(label.copy())
            return dict(images=image, labels=label, scene_name=scene_name, cam_name=cam_name, img_id=img_id,
                        image_shape=im_shape)
        else:
            # get data sample
            # image_path, label_path = self.data_index[idx]
            image_path = self.data_index[idx][0]
            label_path = None if self.test_data == "CLASSIC" or self.test_data == 'cabin' else self.data_index[idx][1]
            img_name = os.path.basename(image_path)
            filetype = os.path.splitext(img_name)[-1]
            file_name = os.path.splitext(img_name)[0] + ".png"

            # base dir
            if self.test_data.upper() == 'BIPED' or self.test_data.upper() == 'BIPEDV2':
                img_dir = os.path.join(self.data_root, 'imgs', 'test')
                gt_dir = os.path.join(self.data_root, 'edge_maps', 'test')
            elif self.test_data.upper() == 'CLASSIC' or self.test_data == 'cabin':
                img_dir = self.data_root
                gt_dir = None
            else:
                img_dir = self.data_root
                gt_dir = self.data_root

            # load data
            if filetype == '.hdf5':
                image = h5py.File(os.path.join(img_dir, image_path), 'r')
                image = image['dataset'][:].astype(np.float32)
                image = np.expand_dims(image, axis=-1).repeat(3, axis=-1)
            else:
                if self.test_data == 'cabin':
                    image = cv2.imread(os.path.join(img_dir, image_path), cv2.IMREAD_COLOR)
                else:
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = image[:self.img_height, :self.img_width]
            if not (self.test_data == "CLASSIC" or self.test_data == 'cabin'):
                label = cv2.imread(label_path, cv2.IMREAD_COLOR)
            else:
                label = None

            im_shape = [image.shape[0], image.shape[1]]
            image, label = self.transform(img=image, gt=label)

            return dict(images=image, labels=label, file_names=file_name, image_shape=im_shape)

    def transform(self, img, gt):
        # gt[gt< 51] = 0 # test without gt discrimination
        if self.test_data == "CLASSIC" or self.test_data == 'cabin':
            img_height = self.img_height
            img_width = self.img_width
            print(
                f"actual size: {img.shape}, target size: {(img_height, img_width,)}")
            # img = cv2.resize(img, (self.img_width, self.img_height))
            img = cv2.resize(img, (img_width, img_height))
            gt = None

        # Make images and labels at least 512 by 512
        elif img.shape[0] < 512 or img.shape[1] < 512:
            img = cv2.resize(img, (self.args.test_img_width, self.args.test_img_height))  # 512
            gt = cv2.resize(gt, (self.args.test_img_width, self.args.test_img_height))  # 512

        # Make sure images and labels are divisible by 2^4=16
        elif img.shape[0] % 16 != 0 or img.shape[1] % 16 != 0:
            img_width = ((img.shape[1] // 16) + 1) * 16
            img_height = ((img.shape[0] // 16) + 1) * 16
            img = cv2.resize(img, (img_width, img_height))
            gt = cv2.resize(gt, (img_width, img_height))
        else:
            img_width = self.args.test_img_width
            img_height = self.args.test_img_height
            img = cv2.resize(img, (img_width, img_height))
            gt = cv2.resize(gt, (img_width, img_height))

        # if self.yita is not None:
        #     gt[gt >= self.yita] = 1
        img = np.array(img, dtype=np.float32)
        # if self.rgb:
        #     img = img[:, :, ::-1]  # RGB->BGR
        img -= self.mean_bgr
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        if self.test_data == "CLASSIC":
            gt = np.zeros((img.shape[:2]))
            gt = torch.from_numpy(np.array([gt])).float()
        else:
            gt = np.array(gt, dtype=np.float32)
            if len(gt.shape) == 3:
                gt = gt[:, :, 0]
            gt /= 255.
            gt = torch.from_numpy(np.array([gt])).float()

        return img, gt


class TrainDataset(Dataset):
    train_modes = ['train', 'test', ]
    dataset_types = ['rgbr', ]
    data_types = ['aug', 'real']

    def __init__(self,
                 data_root,
                 img_height,
                 img_width,
                 mean_bgr,
                 train_mode='train',
                 dataset_type='rgbr',
                 #  is_scaling=None,
                 # Whether to crop image or otherwise resize image to match image height and width.
                 crop_img=False,
                 arg=None
                 ):
        self.data_root = data_root
        self.train_mode = train_mode
        self.dataset_type = dataset_type
        self.data_type = 'aug' if arg.data_augmentation else 'real'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img
        self.arg = arg
        self.two_type = arg.two_type
        self.data_index = self._build_index()

    def _build_index(self):
        assert self.train_mode in self.train_modes, self.train_mode
        assert self.dataset_type in self.dataset_types, self.dataset_type
        assert self.data_type in self.data_types, self.data_type

        data_root = os.path.abspath(self.data_root)
        sample_indices = []
        dataset_inf = dataset_info(self.arg.train_data)
        if self.arg.train_data.lower() == 'biped' or self.arg.train_data.lower() == 'bipedv2':
            images_path = os.path.join(data_root,
                                       'imgs',
                                       self.train_mode,
                                       self.dataset_type,
                                       self.data_type)
            labels_path = os.path.join(data_root,
                                       'edge_maps',
                                       self.train_mode,
                                       self.dataset_type,
                                       self.data_type)

            if self.data_type == 'aug':
                for directory_name in os.listdir(images_path):
                    image_directories = os.path.join(images_path, directory_name)
                    for file_name_ext in os.listdir(image_directories):
                        file_name = os.path.splitext(file_name_ext)[0]
                        sample_indices.append(
                            (os.path.join(images_path, directory_name, file_name + '.jpg'),
                             os.path.join(labels_path, directory_name, file_name + '.png'))
                        )
            else:
                for file_name_ext in os.listdir(images_path):
                    file_name = os.path.splitext(file_name_ext)[0]
                    sample_indices.append(
                        (os.path.join(images_path, file_name + '.jpg'),
                         os.path.join(labels_path, file_name + '.png'))
                    )
        elif self.arg.train_data == 'ML-Hypersim':
            with open(dataset_inf['train_list'], 'r') as f:
                while True:
                    line = f.readline()  # 整行读取数据
                    if not line:
                        break
                    line = line.strip()
                    line_sep = line.split(',')
                    scene_path = os.path.join(self.data_root, line_sep[0], 'images')
                    img_id = '%04d' % int(line_sep[2])
                    sample_indices.append(
                        (os.path.join(scene_path, f'scene_{line_sep[1]}_final_preview',
                                      f'frame.{img_id}.tonemap.jpg'),
                         os.path.join(scene_path, f'scene_{line_sep[1]}_geometry_preview',
                                      f'frame.{img_id}.mesh_edge.png'),
                         os.path.join(scene_path, f'scene_{line_sep[1]}_geometry_hdf5',
                                      f'frame.{img_id}.render_entity_id.hdf5'),
                         line_sep[0], line_sep[1], img_id)
                    )
        else:
            file_path = os.path.join(data_root, self.arg.train_list)
            if self.arg.train_data.lower() == 'bsds':

                with open(file_path, 'r') as f:
                    files = f.readlines()
                files = [line.strip() for line in files]

                pairs = [line.split() for line in files]
                for pair in pairs:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    sample_indices.append(
                        (os.path.join(data_root, tmp_img),
                         os.path.join(data_root, tmp_gt),))
            else:
                with open(file_path) as f:
                    files = json.load(f)
                for pair in files:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    sample_indices.append(
                        (os.path.join(data_root, tmp_img),
                         os.path.join(data_root, tmp_gt),))

        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # load data
        if self.arg.train_data == 'ML-Hypersim':

            crop_size_h = int(768 / resize_scale)
            crop_size_w = int(1024 / resize_scale)
            image_path, label_path_geo, label_path_render, scene_name, cam_name, img_id = self.data_index[idx]
            image_cv2 = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_cv2 = cv2.resize(image_cv2, dsize=(crop_size_w, crop_size_h))
            image = np.array(image_cv2).astype(np.float32)

            label_geo = cv2.imread(label_path_geo, cv2.IMREAD_COLOR)
            label_geo = cv2.resize(label_geo, dsize=(crop_size_w, crop_size_h))
            # label_geo = np.array(label_geo_cv2)  # in color BGR(not RGB)
            # print(label_path_geo)
            condition_blank = (label_geo[:, :, 0] == 255) & (label_geo[:, :, 1] == 255) & (label_geo[:, :, 2] == 255)
            blank = np.where(condition_blank)
            label_geo[blank] = np.array([0, 0, 0])
            image[blank] = np.array([0.0, 0.0, 0.0])

            if self.two_type:
                condition_SE = (label_geo[:, :, 0] >= 150)  # B channel: PE
                PE = np.where(condition_SE)
                condition_SiE = (label_geo[:, :, 1] >= 150)  # G channel: SiE
                SiE = np.where(condition_SiE)
                condition_PE = (label_geo[:, :, 2] >= 150)  # R channel: SE
                SE = np.where(condition_PE)
                label_geo_code = np.zeros([label_geo.shape[0], label_geo.shape[1]], dtype=int)
                label_geo_code[SE] = 1
                label_geo_code[SiE] = 1
                label_geo_code[PE] = 2
                label_geo = label_geo_code

                label = label_geo[:, :, np.newaxis]
                # label_geo_sum = label_geo.sum(2).astype(bool).astype(int)
                # label_geo_argmin = np.argmin(label_geo, axis=2) + 1
                # label_geo = label_geo_sum * label_geo_argmin  # 0 as background, 1 as SiE, 2 as SE, 3 as PE
                # label_geo[np.where(label_geo == 1)] = 2

                # with h5py.File(label_path_render, 'r') as f:
                #     label_render = f["dataset"][:].astype(np.int32)
                #     label_render = find_boundaries_color(label_render, mode='inner')
                #     label_render = label_render[:, :, np.newaxis]
                # # label = np.stack([label_geo, label_render], axis=2).astype(np.float32)
                # label = np.concatenate([label_geo, label_render], axis=2).astype(np.float32)
            else:
                # label = cv2.cvtColor(label_geo, cv2.COLOR_BGR2GRAY).astype(float)
                # label = cv2.imread(label_path_geo, cv2.IMREAD_GRAYSCALE).astype(float)

                label = label_geo.sum(axis=2)
                label = np.clip(label, 0, 1)

                # label[np.where(label >= 50)] = 255
                label = label[:, :, np.newaxis].astype(int)

            image = image.transpose([2, 0, 1])
            label = label.transpose([2, 0, 1])
            image = torch.from_numpy(image.copy()).float()
            label = torch.from_numpy(label.copy())
            return dict(images=image, labels=label, scene_name=scene_name, cam_name=cam_name, img_id=img_id)
        else:
            image_path, label_path = self.data_index[idx]
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            image, label = self.transform(img=image, gt=label)
            return dict(images=image, labels=label)

    def transform(self, img, gt):
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3 and self.arg.train_data != 'ML-Hypersim':
            gt = gt[:, :, 0]
            gt /= 255.  # for DexiNed input and BDCN

        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        i_h, i_w, _ = img.shape
        # data = []
        # if self.scale is not None:
        #     for scl in self.scale:
        #         img_scale = cv2.resize(img, None, fx=scl, fy=scl, interpolation=cv2.INTER_LINEAR)
        #         data.append(torch.from_numpy(img_scale.transpose((2, 0, 1))).float())
        #     return data, gt
        #  400 for BIPEd and 352 for BSDS check with 384
        crop_size = self.img_height if self.img_height == self.img_width else 320  # MDBD=480 BIPED=480/352 BSDS=320

        # for BSDS
        if i_w > crop_size and i_h > crop_size:
            i = random.randint(0, i_h - crop_size)
            j = random.randint(0, i_w - crop_size)
            img = img[i:i + crop_size, j:j + crop_size]
            gt = gt[i:i + crop_size, j:j + crop_size]

        # # for BIPED
        # if np.random.random() > 0.5: #l
        #     h,w = gt.shape
        #     LR_img_size = 256  #l BIPED=256, 240 200 # MDBD= 352 BSDS= 176
        #     i = random.randint(0, h - LR_img_size)
        #     j = random.randint(0, w - LR_img_size)
        #     # if img.
        #     img = img[i:i + LR_img_size , j:j + LR_img_size ]
        #     gt = gt[i:i + LR_img_size , j:j + LR_img_size ]
        #     img = cv2.resize(img, dsize=(crop_size, crop_size),)
        #     gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        else:
            # New addidings
            img = cv2.resize(img, dsize=(crop_size, crop_size))
            gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        # for  BIPED and BSDS
        gt[gt > 0.2] += 0.6  # 0.5 for IPED
        gt = np.clip(gt, 0., 1.)
        # for MDBD
        # gt[gt > 0.1] =1.
        # # gt = np.clip(gt, 0., 1.)
        # # For RCF input
        # # -----------------------------------
        # gt[gt==0]=0.
        # gt[np.logical_and(gt>0.,gt<0.5)] = 2.
        # gt[gt>=0.5]=1.
        #
        # gt = gt.astype('float32')
        # ----------------------------------

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(np.array([gt])).float()
        return img, gt


def canny(image, low_threshold=None, high_threshold=None, mask=None,
          use_quantiles=False):
    """
    low_threshold : float, optional
        Lower bound for hysteresis thresholding (linking edges).
        If None, low_threshold is set to 10% of dtype's max.
    high_threshold : float, optional
        Upper bound for hysteresis thresholding (linking edges).
        If None, high_threshold is set to 20% of dtype's max.
    mask : array, dtype=bool, optional
        Mask to limit the application of Canny to a certain area.


    Returns
    -------
    output : 2D array (image)
        The binary edge map.

    See also
    --------
    skimage.sobel

    Notes
    -----
    The steps of the algorithm are as follows:

    * Smooth the image using a Gaussian with ``sigma`` width.

    * Apply the horizontal and vertical Sobel operators to get the gradients
      within the image. The edge strength is the norm of the gradient.

    * Thin potential edges to 1-pixel wide curves. First, find the normal
      to the edge at each point. This is done by looking at the
      signs and the relative magnitude of the X-Sobel and Y-Sobel
      to sort the points into 4 categories: horizontal, vertical,
      diagonal and antidiagonal. Then look in the normal and reverse
      directions to see if the values in either of those directions are
      greater than the point in question. Use interpolation to get a mix of
      points instead of picking the one that's the closest to the normal.

    * Perform a hysteresis thresholding: first label all points above the
      high threshold as edges. Then recursively label any point above the
      low threshold that is 8-connected to a labeled point as an edge.

    """

    if low_threshold is None:
        low_threshold = 0.1

    if high_threshold is None:
        high_threshold = 0.2

    if mask is None:
        mask = np.ones(image.shape, dtype=bool)

    #
    # Make the eroded mask. Setting the border value to zero will wipe
    # out the image edges for us.
    #
    s = ndi.generate_binary_structure(2, 2)
    eroded_mask = binary_erosion(mask, s, border_value=0)
    eroded_mask = eroded_mask & (magnitude > 0)
    #
    # --------- Find local maxima --------------
    #
    # Assign each point to have a normal of 0-45 degrees, 45-90 degrees,
    # 90-135 degrees and 135-180 degrees.
    #
    local_maxima = np.zeros(image.shape, bool)
    # ----- 0 to 45 degrees ------
    pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
    pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    # Get the magnitudes shifted left to make a matrix of the points to the
    # right of pts. Similarly, shift left and down to get the points to the
    # top right of pts.
    c1 = magnitude[1:, :][pts[:-1, :]]
    c2 = magnitude[1:, 1:][pts[:-1, :-1]]
    m = magnitude[pts]
    w = abs_jsobel[pts] / abs_isobel[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[:-1, :][pts[1:, :]]
    c2 = magnitude[:-1, :-1][pts[1:, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    # ----- 45 to 90 degrees ------
    # Mix diagonal and vertical
    #
    pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel <= abs_jsobel)
    pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel <= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1 = magnitude[:, 1:][pts[:, :-1]]
    c2 = magnitude[1:, 1:][pts[:-1, :-1]]
    m = magnitude[pts]
    w = abs_isobel[pts] / abs_jsobel[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[:, :-1][pts[:, 1:]]
    c2 = magnitude[:-1, :-1][pts[1:, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    # ----- 90 to 135 degrees ------
    # Mix anti-diagonal and vertical
    #
    pts_plus = (isobel <= 0) & (jsobel >= 0) & (abs_isobel <= abs_jsobel)
    pts_minus = (isobel >= 0) & (jsobel <= 0) & (abs_isobel <= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1a = magnitude[:, 1:][pts[:, :-1]]
    c2a = magnitude[:-1, 1:][pts[1:, :-1]]
    m = magnitude[pts]
    w = abs_isobel[pts] / abs_jsobel[pts]
    c_plus = c2a * w + c1a * (1.0 - w) <= m
    c1 = magnitude[:, :-1][pts[:, 1:]]
    c2 = magnitude[1:, :-1][pts[:-1, 1:]]
    c_minus = c2 * w + c1 * (1.0 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    # ----- 135 to 180 degrees ------
    # Mix anti-diagonal and anti-horizontal
    #
    pts_plus = (isobel <= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
    pts_minus = (isobel >= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1 = magnitude[:-1, :][pts[1:, :]]
    c2 = magnitude[:-1, 1:][pts[1:, :-1]]
    m = magnitude[pts]
    w = abs_jsobel[pts] / abs_isobel[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[1:, :][pts[:-1, :]]
    c2 = magnitude[1:, :-1][pts[:-1, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus

    #
    # ---- If use_quantiles is set then calculate the thresholds to use
    #
    if use_quantiles:
        high_threshold = np.percentile(magnitude, 100.0 * high_threshold)
        low_threshold = np.percentile(magnitude, 100.0 * low_threshold)

    #
    # ---- Create two masks at the two thresholds.
    #
    high_mask = local_maxima & (magnitude >= high_threshold)
    low_mask = local_maxima & (magnitude >= low_threshold)

    #
    # Segment the low-mask, then only keep low-segments that have
    # some high_mask component in them
    #
    strel = np.ones((3, 3), bool)
    labels, count = label(low_mask, strel)
    if count == 0:
        return low_mask

    sums = (np.array(ndi.sum(high_mask, labels,
                             np.arange(count, dtype=np.int32) + 1),
                     copy=False, ndmin=1))
    good_label = np.zeros((count + 1,), bool)
    good_label[1:] = sums > 0
    output_mask = good_label[labels]
    return output_mask

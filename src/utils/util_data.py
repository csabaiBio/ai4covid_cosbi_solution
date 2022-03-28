import torch
import numpy as np
from PIL import Image
import cv2
import os
import imutils
import random
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import shift
import pydicom
import pandas as pd


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_img(img_path):
    filename, extension = os.path.splitext(img_path)
    if extension == ".dcm":
        dicom = pydicom.dcmread(img_path)
        img = dicom.pixel_array.astype(float)
        photometric_interpretation = dicom.PhotometricInterpretation
    else:
        img = Image.open(img_path)
        img = np.array(img).astype(float)
        photometric_interpretation = None
    return img, photometric_interpretation


def get_mask(img, mask, value=1):
    mask = mask != value
    img[mask] = 0
    return img


def get_box(img, box, perc_border=.0):
    # Sides
    l_h = box[2] - box[0]
    l_w = box[3] - box[1]
    # Border
    diff = int((abs(l_h - l_w) / 2))
    border = int(perc_border * diff)
    if l_h > l_w:
        max_diff = min(box[1], img.shape[1]-box[3])
        diff = min(max_diff, diff)
        max_border = min(box[0], img.shape[0]-box[2], box[1]-diff, img.shape[1]-box[3]-diff)
        border = min(max_border, border)
        img = img[box[0]-border:box[2]+border, box[1]-diff-border:box[3]+diff+border]
    elif l_w > l_h:
        max_diff = min(box[0], img.shape[0]-box[2])
        diff = min(max_diff, diff)
        max_border = min(box[0]-diff, img.shape[0]-box[2]-diff, box[1], img.shape[1]-box[3])
        border = min(max_border, border)
        img = img[box[0]-diff-border:box[2]+diff+border, box[1]-border:box[3]+border]
    else:
        max_border = min(box[0], img.shape[0]-box[2], box[1], img.shape[1]-box[3])
        border = min(max_border, border)
        img = img[box[0]-border:box[2]+border, box[1]-border:box[3]+border]
    return img


def normalize(img, min_val=None, max_val=None):
    if not min_val:
        min_val = img.min()
    if not max_val:
        max_val = img.max()
    img = (img - min_val) / (max_val - min_val)
    # img -= img.mean()
    # img /= img.std()
    return img


def clahe_transform(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply((img * 255).astype(np.uint8)) / 255
    return img


def elastic_transform(image, alpha_range, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.

   # Arguments
       image: Numpy array with shape (height, width, channels).
       alpha_range: Float for fixed value or [lower, upper] for random value from uniform distribution.
           Controls intensity of deformation.
       sigma: Float, sigma of gaussian filter that smooths the displacement fields.
       random_state: `numpy.random.RandomState` object for generating displacement fields.
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    if np.isscalar(alpha_range):
        alpha = alpha_range
    else:
        alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    # Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result


def augmentation(img):
    # shift
    r = random.randint(0, 100)
    if r > 70:
        shift_perc = 0.1
        r1 = random.randint(-int(shift_perc*img.shape[0]), int(shift_perc*img.shape[0]))
        r2 = random.randint(-int(shift_perc*img.shape[1]), int(shift_perc*img.shape[1]))
        img = shift(img, [r1, r2, 0], mode='nearest')
    # zoom
    r = random.randint(0, 100)
    if r > 70:
        zoom_perc = 0.1
        zoom_factor = random.uniform(1-zoom_perc, 1+zoom_perc)
        img = clipped_zoom(img, zoom_factor=zoom_factor)
    # flip
    r = random.randint(0, 100)
    if r > 70:
        img = cv2.flip(img, 1)
    # rotation
    r = random.randint(0, 100)
    if r > 70:
        max_angle = 15
        r = random.randint(-max_angle, max_angle)
        img = imutils.rotate(img, r)
    # elastic deformation
    r = random.randint(0, 100)
    if r > 70:
        img = elastic_transform(img, alpha_range=[20, 40], sigma=7)
    return img


def loader(img_path, img_dim, mask_path=None, box=None, clahe=False, step="train"):
    # Img
    img, photometric_interpretation = load_img(img_path)
    #print('IMG_LOADED_SUCCESS')
    min_val, max_val = img.min(), img.max()
    # Pathometric Interpretation
    if photometric_interpretation == 'MONOCHROME1':
        img = np.interp(img, (min_val, max_val), (max_val, min_val))
    # To Grayscale
    if img.ndim > 2:
        img = img.mean(axis=2)
    # Filter Mask
    if mask_path:
        mask, _ = load_img(mask_path)
        img = get_mask(img, mask, value=1)
    # Select Box Area
    if box:
        img = get_box(img, box, perc_border=0.5)
    # Resize
    img = cv2.resize(img, (img_dim, img_dim))
    # Normalize
    img = normalize(img, min_val=min_val, max_val=max_val)
    # CLAHE
    if clahe:
        img = clahe_transform(img)
    # To 3 Channels
    img = np.stack((img, img, img), axis=-1)
    if step == "train":
        img = augmentation(img)
    # To Tensor
    img = torch.Tensor(img)
    img = img.permute(2, 0, 1)
    return img


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, classes, img_dir, mask_file, mask_dir, box_file, clahe, step, img_dim):
        'Initialization'
        self.step = step
        self.img_dir = img_dir
        self.data = data
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        # Mask
        if mask_file:
            mask_data = pd.read_excel(mask_file, index_col="img", dtype=list)
            self.masks = {os.path.basename(row[0]): os.path.join(mask_dir, row[1]["mask"]) for row in mask_data.iterrows()}
        else:
            self.masks = None
        # Box
        if box_file:
            box_data = pd.read_excel(box_file, index_col="img", dtype=list)
            # https://stackoverflow.com/questions/65254535/xlrd-biffh-xlrderror-excel-xlsx-file-not-supported
            #df1 = pd.read_excel(
            #        os.path.join(box_file, "img", "aug_latest.xlsm"),
            #        engine='openpyxl',
            #        )
            # just needed to install previous version of xlrd
            self.boxes = {os.path.basename(row[0]): eval(row[1]["box"]) for row in box_data.iterrows()}
        else:
            self.boxes = None
        self.clahe = clahe
        self.img_dim = img_dim

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        row = self.data.iloc[index]
        img_id = row.name
        #print(img_id)
        #print('masks', self.masks)
        #print('boxes', self.boxes.keys())
        if self.masks:
            mask_path = self.masks[os.path.basename(img_id)]
        else:
            mask_path = None
        # load box
        if self.boxes:
            #print('img_id:', os.path.basename(img_id))
            box_id = img_id.strip('.png')
            #box = self.boxes[os.path.basename(img_id)]
            box = self.boxes[os.path.basename(box_id.strip('.png'))]
            ## print('box', box)
        else:
            box = None
        # Load data and get label
        ###img_path = os.path.join(self.img_dir, "%s.dcm" % str(img_id)) ## HARDCODED DICOM
        #print(img_id)
        img_path = os.path.join(self.img_dir, "%s" % str(img_id))
        #print('Image path', img_path)
        x = loader(img_path=img_path, img_dim=self.img_dim, mask_path=mask_path, box=box, clahe=self.clahe, step=self.step)
        try:
            y = row.label
            return x, self.class_to_idx[y], os.path.basename(img_id)
        except:
            return x, -1, os.path.basename(img_id) # nan is encoded as -1


class ClinicalDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, classes, data_file, step):
        'Initialization'
        self.step = step
        #print("HERE", data_file)
        self.clinical_data = pd.read_csv(data_file, index_col=0)
        self.data = data
        #print(self.data.head())
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        #print(self.data, self.class_to_idx)
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        row = self.data.iloc[index]
        id = row.name
        # Load data and get label
        #print( 'In clin data', os.path.basename(id), id, row )
        #print(self.clinical_data.shape)
        x = torch.tensor(self.clinical_data.loc[os.path.basename(id)].astype(float))
        #x = torch.tensor(self.clinical_data.loc[os.path.basename(id.strip('.png'))].astype(float))
        #print('x', x)
        y = row.label
        try:
            return x, self.class_to_idx[y], os.path.basename(id)
        except:
            return x, -1, os.path.basename(id) # nan is encoded as -1


class MultimodalDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dataset_mode_1, dataset_mode_2, classes):
        'Initialization'
        self.dataset_mode_1 = dataset_mode_1
        self.dataset_mode_2 = dataset_mode_2
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset_mode_1)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x1, label, idx = self.dataset_mode_1[index]
        x2, label, idx = self.dataset_mode_2[index]
        return x1, x2, label, idx


class PredictionDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, classes, data, features, step):
        'Initialization'
        self.step = step
        self.data = data
        self.classes = classes
        self.features = features
        self.class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        row = self.data.iloc[index]
        id = row.name
        # Load data and get label
        x = torch.tensor(row[self.features].astype(float))
        y = int(row['True'])
        return x, y, id
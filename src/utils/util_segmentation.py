import numpy as np
from skimage import morphology, color, transform
import os
from tqdm import tqdm
import pydicom

import sys
sys.path.append("/home/abiricz/ai4covid_winners/COVIDCXRChallenge")

from src.utils import util_data

# ADD HOME parent path
HOME = '/home/abiricz/ai4covid_winners/COVIDCXRChallenge/'

def masked_pred(img, mask, alpha=1):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    color_mask[mask == 1] = [0, 0, 1]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked


def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img


def loadData(img_dirs, im_shape):
    """This function loads data preprocessed with `preprocess_JSRT.py`"""
    X = []
    original_shape = []
    img_name = []
    for img_dir in img_dirs:
        print(img_dir)
        for item in tqdm(os.listdir(img_dir)):
            img_name.append(os. path. splitext(item)[0])

            # Load DICOM
        #    dicom = pydicom.dcmread(os.path.join(img_dir, item))
        #    photometric_interpretation = dicom.PhotometricInterpretation
        #    img = dicom.pixel_array.astype(float)

            # MODIFIED LOADER!
            img, photometric_interpretation = util_data.load_img( os.path.join(img_dir, item) )
            
            # Photometric Interpretation
            if photometric_interpretation == 'MONOCHROME1':
                img = np.interp(img, (img.min(), img.max()), (img.max(), img.min()))

            # to grayscale
            if img.ndim > 2:
                img = img.mean(axis=2)
            # Shape
            original_shape.append(img.shape)
            img = transform.resize(img, im_shape)
            # to 1 channel
            img = np.expand_dims(img, -1)
            # Normalize
            img -= img.mean()
            img /= img.std()
            X.append(img)
    X = np.array(X)

    print('### Data loaded')
    print('\t{}'.format(X.shape))
    return X, original_shape, img_name
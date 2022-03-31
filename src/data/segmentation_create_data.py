import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage import exposure, transform
import skimage.measure as measure
import os
from tqdm import tqdm

import src.utils.util_segmentation as util_segmentation


data_dir = "../data/AIforCOVID"
img_dirs = [os.path.join(data_dir, "imgs")] #imgs_test

clinical_data_files = [os.path.join(data_dir, "trainClinData.xls")] #testClinData.xls
y_label = "Prognosis"

# Model path
model_path = './models/segmentation/trained_model.hdf5'
im_shape = (256, 256)

# load clinical data
clinical_data = pd.DataFrame()
for clinical_data_file in clinical_data_files:
    clinical_data = pd.concat([clinical_data, pd.read_excel(clinical_data_file, index_col="ImageFile")])

# Load U-net
UNet = load_model(model_path)

# Bounding-box file
bounding_box_file = os.path.join("./data/processed", "data.xlsx") # data_test.xlsx

# Load test data
X, original_shape, img_name = util_segmentation.loadData(img_dirs, im_shape)

n_test = X.shape[0]
inp_shape = X[0].shape

# For inference standard keras ImageGenerator is used.
test_gen = ImageDataGenerator(rescale=1.)

i = 0
# Bounding Boxes
dx_box = []
sx_box = []
all_box = []
for xx in tqdm(X):
    img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0, 1))
    pred = UNet.predict(np.expand_dims(xx, axis=0))[..., 0].reshape(inp_shape[:2])

    # Binarize masks
    pr = pred > 0.5

    # Remove regions smaller than 2% of the image
    pr = util_segmentation.remove_small_regions(pr, 0.02 * np.prod(im_shape))

    # resize
    pr = transform.resize(pr, original_shape[i])
    # get box for single lungs
    lbl = measure.label(pr)
    props = measure.regionprops(lbl)
    # devo fare singolo
    if len(props) >= 2:
        box_1 = props[0].bbox
        box_2 = props[1].bbox
        if box_1[1] < box_2[1]:
            dx_box.append(list(box_1))
            sx_box.append(list(box_2))
        else:
            dx_box.append(list(box_2))
            sx_box.append(list(box_1))
        # get box for both lungs
        props = measure.regionprops(pr.astype("int64"))
        if len(props) == 1:
            all_box.append(list(props[0].bbox))
        else:
            all_box.append([0, 0, lbl.shape[0], lbl.shape[1]])
    else:
        dx_box.append(None)
        sx_box.append(None)
        all_box.append([0, 0, lbl.shape[0], lbl.shape[1]])

    i += 1
    if i == n_test:
        break

# save excel with boxes
bounding_box = pd.DataFrame(index=img_name)
# bounding_box["dx"] = dx_box
# bounding_box["sx"] = sx_box
bounding_box["box"] = all_box
bounding_box["label"] = clinical_data[y_label]

# dropna
# bounding_box = bounding_box.dropna(subset=["label"])

bounding_box.to_excel(bounding_box_file, index=True, index_label="img")

import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import os
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

import src.utils.util_general as util_general

# Seed Everything
seed = 0
util_general.seed_all(seed)

# Params
label_col = "Prognosis"
group_cols = ["Prognosis", "Hospital"]
test_size = 0.1
val_size = 0.1

# Files and Directories
dest_dir = os.path.join("./data/processed/folds")
data_dir = "../data/AIforCOVID"
clinical_data_files = [os.path.join(data_dir, "trainClinData.xls")]

# load clinical data
clinical_data = pd.DataFrame()
for clinical_data_file in clinical_data_files:
    clinical_data_r = pd.read_excel(clinical_data_file, index_col="ImageFile")
    clinical_data_r.index = [os.path.join("imgs", x) for x in clinical_data_r.index]
    clinical_data = pd.concat([clinical_data, clinical_data_r])

# K-Folds CV
fold_data = {}
train, test = train_test_split(clinical_data, test_size=test_size, stratify=clinical_data[group_cols], random_state=0)
train, val = train_test_split(train, test_size=val_size, stratify=train[group_cols], random_state=0)
fold_data['train'] = train.index.to_list()
fold_data['val'] = val.index.to_list()
fold_data['test'] = test.index.to_list()

# all.txt
with open(os.path.join(dest_dir, 'all.txt'), 'w') as file:
    file.write("img label\n")
    for img in clinical_data.index:
        label = "%s\n" % clinical_data.loc[img, label_col]
        row = "%s %s" % (img, label)
        file.write(row)

# create split dir
steps = ['train', 'val', 'test']
# .txt
for step in steps:
    with open(os.path.join(dest_dir, '%s.txt' % step), 'w') as file:
        file.write("img label\n")
        for img in tqdm(fold_data[step]):
            label = "%s\n" % clinical_data.loc[img, label_col]
            row = "%s %s" % (img, label)
            file.write(row)

# submission
clinical_data_files = [os.path.join(data_dir, "testClinData.xls")]

# load clinical data
clinical_data = pd.DataFrame()
for clinical_data_file in clinical_data_files:
    clinical_data_r = pd.read_excel(clinical_data_file, index_col="ImageFile")
    clinical_data_r.index = [os.path.join("imgs_test", x) for x in clinical_data_r.index]
    clinical_data = pd.concat([clinical_data, clinical_data_r])

with open(os.path.join(dest_dir, 'submission.txt'), 'w') as file:
    file.write("img label\n")
    for img in clinical_data.index:
        label = "%s\n" % clinical_data.loc[img, label_col]
        row = "%s %s" % (img, label)
        file.write(row)

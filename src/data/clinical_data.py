import pandas as pd
import os
import pickle
from sklearn.preprocessing import MinMaxScaler

import src.utils.util_general as util_general

data_dir = "../data/AIforCOVID"
preprocessing_dir = os.path.join("./data/interim/preprocessing", "clinical")
util_general.create_dir(preprocessing_dir)
save_dir = "./data/processed"
y_label = "Prognosis"

clinical_data_files = [os.path.join(data_dir, "trainClinData.xls")]
drop_cols = ["Row_number", "Death"]

# load clinical data
clinical_data = pd.DataFrame()
for clinical_data_file in clinical_data_files:
    clinical_data = pd.concat([clinical_data, pd.read_excel(clinical_data_file, index_col="ImageFile")])

# Drop cols
clinical_data = clinical_data.drop(drop_cols, axis=1)

# Fill NA
cat_cols = ["Hospital"]
discrete_cols = ['Age', 'Sex', 'Positivity at admission', 'Temp_C', 'DaysFever', 'Cough', 'DifficultyInBreathing', 'WBC',
            'RBC', 'Fibrinogen', 'Glucose', 'LDH', 'D-dimer', 'Ox_percentage', 'PaO2', 'SaO2', 'PaCO2', 'pH',
            'CardiovascularDisease', 'IschemicHeartDisease', 'AtrialFibrillation', 'HeartFailure', 'Ictus',
            'HighBloodPressure', 'Diabetes', 'Dementia', 'BPCO', 'Cancer', 'Chronic Kidney disease',
            'RespiratoryFailure', 'Obesity', "Position"]
cont_cols = ['CRP', 'PCT', 'INR']

# Fill NA
na_dict = {}
for col in clinical_data:
    if col != y_label:
        if col in discrete_cols:
            mean = clinical_data[col].mean()
            clinical_data[col] = clinical_data[col].fillna(int(mean))
            na_dict[col] = int(mean)
        elif col in cont_cols:
            mean = clinical_data[col].mean()
            clinical_data[col] = clinical_data[col].fillna(round(mean, 2))
            na_dict[col] = round(mean, 2)
        elif col in cont_cols:
            common = clinical_data[col].mode().item()
            clinical_data[col] = clinical_data[col].fillna(common)
            na_dict[col] = common
# save na info
with open(os.path.join(preprocessing_dir, 'na.pkl'), 'wb') as handle:
    pickle.dump(na_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Drop y
clinical_data = clinical_data.drop(y_label, axis=1)

# Onehot
for col in cat_cols:
    one_hot = pd.get_dummies(clinical_data[col])
    clinical_data = clinical_data.drop(col, axis=1)
    clinical_data = clinical_data.join(one_hot)

# Scaler
scaler = MinMaxScaler()
clinical_data = pd.DataFrame(scaler.fit_transform(clinical_data), index=clinical_data.index, columns=clinical_data.columns)
# save scaler
with open(os.path.join(preprocessing_dir, 'scaler.pkl'), 'wb') as handle:
    pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save
clinical_data.to_csv(os.path.join(save_dir, "clinical_data.csv"), index=True)

# Test
clinical_data_files = [os.path.join(data_dir, "testClinData.xls")]

# load clinical data
clinical_data = pd.DataFrame()
for clinical_data_file in clinical_data_files:
    clinical_data = pd.concat([clinical_data, pd.read_excel(clinical_data_file, index_col="ImageFile")])

# Drop cols
clinical_data = clinical_data.drop(drop_cols, axis=1)

# Fill NA
for col in clinical_data:
    if col != y_label:
        if col in discrete_cols:
            clinical_data[col] = clinical_data[col].fillna(na_dict[col])
        elif col in cont_cols:
            clinical_data[col] = clinical_data[col].fillna(na_dict[col])
        elif col in cont_cols:
            clinical_data[col] = clinical_data[col].fillna(na_dict[col])

# Drop y
clinical_data = clinical_data.drop(y_label, axis=1)

# Onehot
clinical_data = clinical_data.drop('Hospital', axis=1)
for c in ["A", "B", "C", "D", "E"]:
    clinical_data[c] = 0
clinical_data["F"] = 1

# Scaler
clinical_data = pd.DataFrame(scaler.transform(clinical_data), index=clinical_data.index, columns=clinical_data.columns)

# Save
clinical_data.to_csv(os.path.join(save_dir, "clinical_data_test.csv"), index=True)

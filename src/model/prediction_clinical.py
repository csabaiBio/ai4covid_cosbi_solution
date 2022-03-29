import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import os

filepath = os.path.abspath(__file__)
HOME = '/'.join( filepath.split('/')[:-3] ) + '/'
sys.path.append(HOME)


import torch
import pandas as pd
import os
import yaml

import src.utils.util_general as util_general
import src.utils.util_data as util_data
import src.utils.util_model as util_model

# Configuration file
args = util_general.get_args()
args.cfg_file = HOME+"configs/clinical.yaml"
with open(os.path.join(args.cfg_file)) as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

# Seed everything
util_general.seed_all(cfg['seed'])

# Parameters
exp_name = cfg['exp_name']
classes = cfg['data']['classes']
model_name = cfg['model']['model_name']

# Device
device = torch.device(cfg['device']['cuda_device'] if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
print(device)


# Files and Directories
model_dir = cfg['data']['model_dir']
model_dir_exp = os.path.join(model_dir, exp_name)
report_dir = cfg['data']['report_dir']
report_dir_exp = os.path.join(report_dir, exp_name)
table_dir = os.path.join(report_dir_exp, "tables")
prediction_dir = os.path.join(table_dir, "prediction")
util_general.create_dir(prediction_dir)

# Data Loaders
fold_data = { step: pd.read_csv( os.path.join(cfg['data']['fold_dir'], '%s.txt' % step), 
                                delimiter=" ", index_col=0 
                              ) 
                    for step in ['train', 'val', 'test', 'submission'] }
datasets = {    step: util_data.ClinicalDataset(
                    data=fold_data[step], 
                    classes=classes,
                    data_file=cfg['data']['data_file'], step="test") 
             for step in ['train', 'val', 'test', 'submission']}
data_loaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
                'val': torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
                'test': torch.utils.data.DataLoader(datasets['test'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
                'submission': torch.utils.data.DataLoader(datasets['submission'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker)}

# Split
#for step in ['train', 'val', 'test', 'submission']:
for step in ['submission']:
    prediction_file = os.path.join(prediction_dir, "prediction_%s.xlsx" % step)
    probability_file = os.path.join(prediction_dir, "probability_%s.xlsx" % step)

    # Predict Models
    results_frame = pd.DataFrame()
    probs_frame = pd.DataFrame()

    # Model
    print("%s%s%s" % ("*" * 50, model_name, "*" * 50))
    model = torch.load(os.path.join(model_dir_exp, "%s.pt" % model_name), map_location=device)
    model = model.to(device)

    # Prediction
    predictions, probabilities, truth = util_model.predict(model, data_loaders[step], device)

    # Update report
    results_frame[model_name] = pd.Series(predictions)
    for i in range(len(classes)):
        probs_frame["%s_%i" % (model_name, i)] = pd.Series({x: probs[i] for x, probs in probabilities.items()})

    # Ground Truth
    results_frame["True"] = pd.Series(truth)
    probs_frame["True"] = pd.Series(truth)

    # Save Results
    results_frame.to_excel(prediction_file, index=True)
    probs_frame.to_excel(probability_file, index=True)

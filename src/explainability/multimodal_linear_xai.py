import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import pandas as pd
import os
import collections
import pickle
import yaml

import src.utils.util_general as util_general

# Configuration file
args = util_general.get_args()
args.cfg_file = "./configs/multimodal.yaml"
with open(os.path.join(args.cfg_file)) as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
cfg_modes = {}
for mode, mode_cfg_file in cfg["mode_cfg"].items():
    with open(os.path.join(mode_cfg_file)) as file:
        cfg_modes[mode] = yaml.load(file, Loader=yaml.FullLoader)

# Seed everything
util_general.seed_all(cfg['seed'])

# Parameters
exp_name = cfg['exp_name']
classes = cfg['data']['classes']
model_names = cfg['model']['model_names']

# Files and Directories
model_dir_discrete = os.path.join(cfg['linear']['model_dir'], "discrete")
model_dir_probs = os.path.join(cfg['linear']['model_dir'], "probs")
report_dir = os.path.join(cfg["data"]["report_dir"], "xai")
report_dir_exp = os.path.join(report_dir, exp_name)
table_dir = os.path.join(report_dir_exp, "tables")
util_general.create_dir(table_dir)
report_file_discrete = os.path.join(table_dir, "jointlate_discrete_xai.xlsx")
report_file_probs = os.path.join(table_dir, "jointlate_probs_xai.xlsx")

# Weight Interpretation
results_discrete = collections.defaultdict(lambda: [])
results_probs = collections.defaultdict(lambda: [])
discrete_features = [model_name for mode in model_names for model_id, model_name in model_names[mode].items()]
class_labels = [1]
prob_features = ["%s_%i" % (model_name, c) for mode in model_names for model_id, model_name in model_names[mode].items() for c in class_labels]
model_name = ";".join([";".join(sorted([model_name for model_id, model_name in cfg['model']['model_names'][mode].items()])) for mode in cfg['model']['model_names']])

# Load Model
with open(os.path.join(model_dir_discrete, '%s.pkl' % model_name), 'rb') as handle:
    regressor = pickle.load(handle)
# Weights
results_discrete["Weight"] = list(regressor.coef_)

# Linear Regression Discrete
with open(os.path.join(model_dir_probs, '%s.pkl' % model_name), 'rb') as handle:
    regressor = pickle.load(handle)
# Weights
results_probs["Weight"] = list(regressor.coef_)

# Save Results Discrete
results_frame_dicrete = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results_discrete.items()]))
results_frame_dicrete.insert(loc=0, column='model', value=discrete_features)
results_frame_dicrete.to_excel(report_file_discrete, index=False)

# Save Results Probs
results_frame_probs = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results_probs.items()]))
results_frame_probs.insert(loc=0, column='model', value=prob_features)
results_frame_probs.to_excel(report_file_probs, index=False)

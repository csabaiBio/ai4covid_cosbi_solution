import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
from tqdm import tqdm
import pandas as pd
import os
import collections
import itertools
from sklearn.linear_model import LinearRegression
import yaml
import pickle

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
# Mode 1
model_list_mode_1 = cfg_modes['mode_1']['model']['model_list']
k_list = [1, 2, 3]
# Mode 2
model_name_mode_2 = cfg_modes['mode_2']['model']['model_name']

# Files and Directories
model_dir_discrete = os.path.join(cfg['linear']['model_dir'], "discrete")
util_general.create_dir(model_dir_discrete)
model_dir_probs = os.path.join(cfg['linear']['model_dir'], "probs")
util_general.create_dir(model_dir_probs)
report_dir = cfg['data']['report_dir']
report_dir_exp = os.path.join(report_dir, exp_name)
table_dir = os.path.join(report_dir_exp, "tables")
util_general.create_dir(table_dir)
report_file_discrete = os.path.join(table_dir, "jointlate_discrete.xlsx")
report_file_probs = os.path.join(table_dir, "jointlate_probs.xlsx")

# All K combinations
results_discrete = collections.defaultdict(lambda: [])
results_probs = collections.defaultdict(lambda: [])
model_name_list = ["%s;%s" % (";".join(sorted(comb)), model_name_mode_2) for k in k_list for comb in itertools.combinations(model_list_mode_1, k)]
k_model_list = [k for k in k_list for comb in itertools.combinations(model_list_mode_1, k)]

# Load predictions
predictions = collections.defaultdict(lambda: pd.DataFrame())
probabilities = collections.defaultdict(lambda: pd.DataFrame())
for mode in cfg_modes:
    predictions_mode = {step: pd.read_excel(os.path.join(cfg_modes[mode]['data']['report_dir'], cfg_modes[mode]['exp_name'], "tables", "prediction", "prediction_%s.xlsx" % step), index_col=0) for step in ["train", "test"]}
    probabilities_mode = {step: pd.read_excel(os.path.join(cfg_modes[mode]['data']['report_dir'], cfg_modes[mode]['exp_name'], "tables", "prediction", "probability_%s.xlsx" % step), index_col=0) for step in ["train", "test"]}
    for step in ["train", "test"]:
        predictions[step] = pd.concat([predictions[step], predictions_mode[step]], axis=1)
        probabilities[step] = pd.concat([probabilities[step], probabilities_mode[step]], axis=1)
        # Remove duplicated True
        predictions[step] = predictions[step].loc[:, ~predictions[step].columns.duplicated()]
        probabilities[step] = probabilities[step].loc[:, ~probabilities[step].columns.duplicated()]

for k in k_list:
    print("k=%i" % k)

    for comb in tqdm(itertools.combinations(model_list_mode_1, k)):
        model_name = "%s;%s" % (";".join(sorted(comb)), model_name_mode_2)

        # Linear Regression Discrete
        discrete_features = list(comb) + [model_name_mode_2]
        regressor = LinearRegression()
        regressor.fit(predictions["train"][discrete_features], predictions["train"]["True"])
        # Save model
        with open(os.path.join(model_dir_discrete, '%s.pkl' % model_name), 'wb') as handle:
            pickle.dump(regressor, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Test
        y_pred = regressor.predict(predictions["test"][discrete_features]).round()
        # Accuracy
        comb_results = (y_pred == predictions["test"]["True"]) * 1
        results_discrete["ACC"].append(comb_results.sum() / len(comb_results))
        for cat in predictions["test"]["True"].unique():
            cat_comb_results = comb_results[predictions["test"]["True"] == cat]
            results_discrete["ACC %s" % classes[cat]].append(cat_comb_results.sum() / len(cat_comb_results))

        # Linear Regression Probs
        class_labels = [1]
        prob_features = ["%s_%i" % (model, i) for model in comb for i in class_labels] + ["%s_%i" % (model_name_mode_2, i) for i in class_labels]
        regressor = LinearRegression()
        regressor.fit(probabilities["train"][prob_features], probabilities["train"]["True"])
        # Save model
        with open(os.path.join(model_dir_probs, '%s.pkl' % model_name), 'wb') as handle:
            pickle.dump(regressor, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Test
        y_pred = regressor.predict(probabilities["test"][prob_features]).round()
        # Accuracy
        comb_results = (y_pred == probabilities["test"]["True"]) * 1
        results_probs["ACC"].append(comb_results.sum() / len(comb_results))
        for cat in probabilities["test"]["True"].unique():
            cat_comb_results = comb_results[probabilities["test"]["True"] == cat]
            results_probs["ACC %s" % classes[cat]].append(cat_comb_results.sum() / len(cat_comb_results))

# Save Results Discrete
results_frame_dicrete = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results_discrete.items()]))
results_frame_dicrete.insert(loc=0, column='k', value=k_model_list)
results_frame_dicrete.insert(loc=0, column='model', value=model_name_list)
results_frame_dicrete.to_excel(report_file_discrete, index=False)

# Save Results Probs
results_frame_probs = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results_probs.items()]))
results_frame_probs.insert(loc=0, column='k', value=k_model_list)
results_frame_probs.insert(loc=0, column='model', value=model_name_list)
results_frame_probs.to_excel(report_file_probs, index=False)

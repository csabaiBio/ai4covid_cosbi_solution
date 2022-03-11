import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
from tqdm import tqdm
import pandas as pd
import os
import collections
import itertools
import numpy as np
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
div_measures = ["Q", "Correlation", "Disagreement", "Double-fault", "Kappa"]
# Mode 1
model_list_mode_1 = cfg_modes['mode_1']['model']['model_list']
k_list = [1, 2, 3]
# Mode 2
model_name_mode_2 = cfg_modes['mode_2']['model']['model_name']

# Files and Directories
report_dir = cfg['data']['report_dir']
report_dir_exp = os.path.join(report_dir, exp_name)
table_dir = os.path.join(report_dir_exp, "tables")
util_general.create_dir(table_dir)

# Split
for step in ['test']:
    report_file = os.path.join(table_dir, "optimization_%s.xlsx" % step)

    # All K combinations
    results = collections.defaultdict(lambda: [])
    model_name_list = ["%s;%s" % (";".join(sorted(comb)), model_name_mode_2) for k in k_list for comb in itertools.combinations(model_list_mode_1, k)]
    k_model_list = [k for k in k_list for comb in itertools.combinations(model_list_mode_1, k)]

    # Load predictions
    prediction = pd.DataFrame()
    for mode in cfg_modes:
        predictions_mode = pd.read_excel(os.path.join(cfg_modes[mode]['data']['report_dir'], cfg_modes[mode]['exp_name'], "tables", "prediction", "prediction_%s.xlsx" % step), index_col=0)
        prediction = pd.concat([prediction, predictions_mode], axis=1)
        # Remove duplicated True
        prediction = prediction.loc[:, ~prediction.columns.duplicated()]

    for k in k_list:
        print("k=%i" % k)

        for comb in tqdm(itertools.combinations(model_list_mode_1, k)):
            discrete_features = list(comb) + [model_name_mode_2]
            # Accuracy
            comb_results = (prediction[discrete_features].mode(axis=1)[0] == prediction["True"]) * 1
            results["ACC"].append(comb_results.sum() / len(comb_results))
            for cat in prediction["True"].unique():
                cat_comb_results = comb_results[prediction["True"] == cat]
                results["ACC %s" % classes[cat]].append(cat_comb_results.sum() / len(cat_comb_results))

            # Pairwise diversity measures
            measure_dict = collections.defaultdict(lambda: [])
            for model_1, model_2 in itertools.combinations(discrete_features, 2):
                N_11 = len(prediction[(prediction[model_1] == prediction["True"]) & (prediction[model_2] == prediction["True"])])
                N_10 = len(prediction[(prediction[model_1] == prediction["True"]) & (prediction[model_2] != prediction["True"])])
                N_01 = len(prediction[(prediction[model_1] != prediction["True"]) & (prediction[model_2] == prediction["True"])])
                N_00 = len(prediction[(prediction[model_1] != prediction["True"]) & (prediction[model_2] != prediction["True"])])
                omega_1 = (N_11 + N_00) / (N_11 + N_10 + N_01 + N_00)
                omega_2 = ((N_11 + N_10) * (N_11 + N_01) + (N_01 + N_00) * (N_10 + N_00)) / (N_11 + N_10 + N_01 + N_00)**2
                q = (((N_11 * N_00) - (N_01 * N_10)) / ((N_11 * N_00) + (N_01 * N_10)))
                corr = ((N_11 * N_00) - (N_01 * N_10)) / np.sqrt((N_11 + N_10) * (N_01 + N_00) * (N_11 + N_01) * (N_10 + N_00))
                dis = (N_01 + N_10) / (N_11 + N_10 + N_01 + N_00)
                kappa = (omega_1 - omega_2) / (1 - omega_2)
                df = N_00 / (N_11 + N_10 + N_01 + N_00)
                measure_dict["Q"].append(q)
                measure_dict["Correlation"].append((corr + 1) / 2)
                measure_dict["Disagreement"].append((dis - 1) * (-1))
                measure_dict["Kappa"].append(kappa)
                measure_dict["Double-fault"].append(df)
            # Aggregate
            for div in measure_dict:
                results["DIV %s" % div].append(((2 / ((k+1) * ((k+1) - 1))) * sum(measure_dict[div])))

    # Save Results
    results_frame = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
    # Function
    for div in div_measures:
        f = (1 - results_frame["ACC"]) ** 2 + (1 - results_frame["DIV %s" % div]) ** 2
        results_frame.insert(loc=0, column="F ACC & %s" % div, value=f)
    # Model comb
    results_frame.insert(loc=0, column='k', value=k_model_list)
    results_frame.insert(loc=0, column='model', value=model_name_list)
    results_frame.to_excel(report_file, index=False)

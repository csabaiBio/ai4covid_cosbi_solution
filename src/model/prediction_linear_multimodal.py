import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import os

filepath = os.path.abspath(__file__)
HOME = '/'.join( filepath.split('/')[:-3] ) + '/'
sys.path.append(HOME)

import pandas as pd
import os
import collections
import yaml
import pickle

import src.utils.util_general as util_general

# Configuration file
args = util_general.get_args()
args.cfg_file = HOME+"configs/multimodal.yaml"
with open(os.path.join(os.path.join(args.cfg_file))) as file:
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
discrete_features = [model_name for mode in cfg['model']['model_names'] for model_id, model_name in cfg['model']['model_names'][mode].items()]
class_labels = [1]
prob_features = ["%s_%i" % (model, i) for model in discrete_features for i in class_labels]
linear_model_name = ";".join([";".join(sorted([model_name for model_id, model_name in cfg['model']['model_names'][mode].items()])) for mode in cfg['model']['model_names']])

# Files and Directories
model_dir = cfg['linear']['model_dir']
model_dir_discrete = os.path.join(model_dir, "discrete")
model_dir_probs = os.path.join(model_dir, "probs")
report_dir = cfg['data']['report_dir']
report_dir_exp = os.path.join(report_dir, exp_name)
table_dir = os.path.join(report_dir_exp, "tables")
prediction_dir = os.path.join(table_dir, "prediction")
util_general.create_dir(prediction_dir)

# Split
#for step in ['train', 'val', 'test']:#, 'submission']: ## NEED TO PREPARE SUBMISSION FILE!
for step in ['submission']: ## NEED TO PREPARE SUBMISSION FILE!
    prediction_file = os.path.join(prediction_dir, "prediction_%s.xlsx" % step)
    probability_file = os.path.join(prediction_dir, "probability_%s.xlsx" % step)

    print('Looking for files:', prediction_file, probability_file)
    # Predict Models
    results_discrete = pd.DataFrame()
    results_probs = pd.DataFrame()

    # Load predictions
    prediction = pd.DataFrame()
    probability = pd.DataFrame()
    for mode in cfg_modes:
        print('mode', mode)
        predictions_mode = pd.read_excel(os.path.join(cfg_modes[mode]['data']['report_dir'], cfg_modes[mode]['exp_name'], "tables", "prediction", "prediction_%s.xlsx" % step), index_col=0)
        probabilities_mode = pd.read_excel(os.path.join(cfg_modes[mode]['data']['report_dir'], cfg_modes[mode]['exp_name'], "tables", "prediction", "probability_%s.xlsx" % step), index_col=0)

        prediction = pd.concat([prediction, predictions_mode], axis=1)
        probability = pd.concat([probability, probabilities_mode], axis=1)
        # Remove duplicated True
        prediction = prediction.loc[:, ~prediction.columns.duplicated()]
        probability = probability.loc[:, ~probability.columns.duplicated()]
        #print('probdf:', probability.head())

    feature_names_in = [ 'vgg11_bn',  'densenet121', 'mlp_1' ]
    prob_feature_names_in = [ 'vgg11_bn_1', 'densenet121_1', 'mlp_1_1']
    # Linear Model Test
    with open(os.path.join(model_dir_discrete, '%s.pkl' % linear_model_name), 'rb') as handle:
        regressor = pickle.load(handle)
    #results_discrete[linear_model_name] = pd.Series(abs(regressor.predict(prediction[regressor.feature_names_in_]).round()), index=prediction.index)
    results_discrete[linear_model_name] = pd.Series(abs(regressor.predict(prediction[feature_names_in]).round()), index=prediction.index)

    with open(os.path.join(model_dir_probs, '%s.pkl' % linear_model_name), 'rb') as handle:
        regressor = pickle.load(handle)
    #results_probs[linear_model_name] = pd.Series(abs(regressor.predict(probability[regressor.feature_names_in_]).round()), index=probability.index)
    results_probs[linear_model_name] = pd.Series(abs(regressor.predict(probability[prob_feature_names_in]).round()), index=probability.index)
    
    # Ground Truth
    results_discrete["True"] = prediction['True']
    results_probs["True"] = probability['True']
    # Save Results
    results_discrete.to_excel(prediction_file, index=True)
    results_probs.to_excel(probability_file, index=True)
    
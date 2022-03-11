import sys;print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import torch
import pandas as pd
import os
import yaml
from tqdm import tqdm
from captum.attr import IntegratedGradients, DeepLift, NoiseTunnel, FeatureAblation

import src.utils.util_general as util_general
import src.utils.util_data as util_data
import src.utils.util_xai as util_xai

# Configuration file
args = util_general.get_args()
args.cfg_file = "./configs/clinical.yaml"
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
model_dir_exp = os.path.join(cfg['data']['model_dir'], exp_name)
report_dir_exp = os.path.join(cfg["data"]["report_dir"], "xai", exp_name)
table_dir = os.path.join(report_dir_exp, "tables")
util_general.create_dir(table_dir)
figure_dir = os.path.join(report_dir_exp, "figures")
util_general.create_dir(figure_dir)

# Data Loaders
fold_data = {step: pd.read_csv(os.path.join(cfg['data']['fold_dir'], '%s.txt' % step), delimiter=" ", index_col=0) for step in ['train', 'val', 'test', 'submission']}
datasets = {step: util_data.ClinicalDataset(data=fold_data[step], classes=classes, data_file=cfg['data']['data_file'], step="test") for step in ['train', 'val', 'test', 'submission']}
data_loaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
                'val': torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
                'test': torch.utils.data.DataLoader(datasets['test'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
                'submission': torch.utils.data.DataLoader(datasets['test'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker)}

# Split
for step in ['submission']:
    xai_ig_file = os.path.join(table_dir, "clinical_xai_IntegratedGradients_%s.xlsx" % step)
    xai_ig_nt_file = os.path.join(table_dir, "clinical_xai_NoiseTunnel_%s.xlsx" % step)
    xai_dl_file = os.path.join(table_dir, "clinical_xai_DeepLift_%s.xlsx" % step)
    xai_fa_file = os.path.join(table_dir, "clinical_xai_FeatureAblation_%s.xlsx" % step)
    figures_step_dir = os.path.join(figure_dir, step)
    util_general.create_dir(figures_step_dir)
    ig_figure_step_dir = os.path.join(figures_step_dir, "IntegratedGradients")
    util_general.create_dir(ig_figure_step_dir)
    ig_nt_figure_step_dir = os.path.join(figures_step_dir, "NoiseTunnel")
    util_general.create_dir(ig_nt_figure_step_dir)
    dl_figure_step_dir = os.path.join(figures_step_dir, "DeepLift")
    util_general.create_dir(dl_figure_step_dir)
    fa_figure_step_dir = os.path.join(figures_step_dir, "FeatureAblation")
    util_general.create_dir(fa_figure_step_dir)

    # Model
    print("%s%s%s" % ("*" * 50, model_name, "*" * 50))
    model = torch.load(os.path.join(model_dir_exp, "%s.pt" % model_name), map_location=device)
    model = model.to(device)
    model.eval()

    # XAI Algorithms
    ig = IntegratedGradients(model)
    ig_nt = NoiseTunnel(ig)
    dl = DeepLift(model)
    fa = FeatureAblation(model)

    results_ig = {}
    results_ig_nt = {}
    results_dl = {}
    results_fa = {}
    for inputs, labels, file_names in tqdm(data_loaders[step]):
        ig_attributions = ig.attribute(inputs.float(), target=labels)
        ig_nt_attributions = ig_nt.attribute(inputs.float(), target=labels)
        dl_attributions = dl.attribute(inputs.float(), target=labels)
        fa_attributions = fa.attribute(inputs.float(), target=labels)

        for file_name, ig_attr, ig_nt_attr, dl_attr, fa_attr in zip(file_names, ig_attributions, ig_nt_attributions, dl_attributions, fa_attributions):
            file_name = os.path.splitext(file_name)[0]
            results_ig[file_name] = ig_attr.tolist()
            results_ig_nt[file_name] = ig_nt_attr.tolist()
            results_dl[file_name] = dl_attr.tolist()
            results_fa[file_name] = fa_attr.tolist()

    # Save Results
    results_frame_ig = pd.DataFrame.from_dict(results_ig, orient='index', columns=datasets[step].clinical_data.columns.to_list())
    results_frame_ig.to_excel(xai_ig_file, index=True)
    results_frame_ig_nt = pd.DataFrame.from_dict(results_ig_nt, orient='index', columns=datasets[step].clinical_data.columns.to_list())
    results_frame_ig_nt.to_excel(xai_ig_nt_file, index=True)
    results_frame_dl = pd.DataFrame.from_dict(results_dl, orient='index', columns=datasets[step].clinical_data.columns.to_list())
    results_frame_dl.to_excel(xai_dl_file, index=True)
    results_frame_fa = pd.DataFrame.from_dict(results_fa, orient='index', columns=datasets[step].clinical_data.columns.to_list())
    results_frame_fa.to_excel(xai_fa_file, index=True)

    # Plot Importance
    for patient, row in tqdm(results_frame_ig.iterrows()):
        util_xai.plot_feature_importance(importance=row, figures_dir=ig_figure_step_dir, file_name=patient)
    for patient, row in tqdm(results_frame_ig_nt.iterrows()):
        util_xai.plot_feature_importance(importance=row, figures_dir=ig_nt_figure_step_dir, file_name=patient)
    for patient, row in tqdm(results_frame_dl.iterrows()):
        util_xai.plot_feature_importance(importance=row, figures_dir=dl_figure_step_dir, file_name=patient)
    for patient, row in tqdm(results_frame_fa.iterrows()):
        util_xai.plot_feature_importance(importance=row, figures_dir=fa_figure_step_dir, file_name=patient)

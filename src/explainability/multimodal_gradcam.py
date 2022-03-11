import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import pandas as pd
import os
import torch
import yaml
from tqdm import tqdm
import collections

import src.utils.util_general as util_general
import src.utils.util_data as util_data
import src.utils.util_xai as util_xai


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
exp_name = cfg_modes['mode_1']['exp_name']
classes = cfg['data']['classes']
model_names = cfg['model']['model_names']

# Device
device = torch.device("cpu") #torch.device(cfg['device']['cuda_device'] if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
print(device)

# Files and Directories
model_dir_exp = os.path.join(cfg['data']['model_dir'], exp_name)

report_dir = os.path.join(cfg["data"]["report_dir"], "xai")
report_dir_exp = os.path.join(report_dir, exp_name)
table_dir = os.path.join(report_dir_exp, "tables")
figure_dir = os.path.join(report_dir_exp, "figures")
xai_dir = os.path.join(figure_dir, "gradcam")
util_general.create_dir(xai_dir)

# Load weights Discrete
weights_discrete = pd.read_excel(os.path.join("./reports/xai/multimodal/tables", "jointlate_discrete_xai.xlsx"), index_col="model")
weights_discrete = weights_discrete.loc[model_names["mode_1"].values(), "Weight"]
weights_discrete /= weights_discrete.sum()

# Load Weights Probs
class_labels = [1]
weights_probs = pd.read_excel(os.path.join("./reports/xai/multimodal/tables", "jointlate_probs_xai.xlsx"), index_col="model")
weights_probs = weights_probs.loc[["%s_%i" % (model_name, c) for model_name in model_names["mode_1"].values() for c in class_labels], "Weight"]
weights_probs /= weights_probs.sum()

# Data Loaders
fold_data = {step: pd.read_csv(os.path.join(cfg['data']['fold_dir'], '%s.txt' % step), delimiter=" ", index_col=0) for step in ['train', 'val', 'test', 'submission']}
datasets = {step: util_data.Dataset(data=fold_data[step], classes=classes, img_dir=cfg_modes['mode_1']['data']['img_dir'], mask_file=cfg_modes['mode_1']['data']['mask_file'], mask_dir=cfg_modes['mode_1']['data']['mask_dir'], box_file=cfg_modes['mode_1']['data']['box_file'], clahe=cfg_modes['mode_1']['data']['clahe'], step="test", img_dim=cfg_modes['mode_1']['data']['img_dim']) for step in ['train', 'val', 'test', 'submission']}

for step in ["submission"]:
    # Dir
    xai_step_dir = os.path.join(xai_dir, step)
    util_general.create_dir(xai_step_dir)

    # Images
    images, labels, file_names, raw_images = util_xai.load_images(datasets=datasets, step=step)
    images = torch.stack(images).to(device)

    #####
    #images = images[:20]
    #labels = labels[:20]
    #file_names = file_names[:20]
    #raw_images = raw_images[:20]
    #####

    # Dir
    for model_id, model_name in model_names['mode_1'].items():
        xai_model_step_dir = os.path.join(xai_step_dir, model_name)
        util_general.create_dir(xai_model_step_dir)
    model_name = ";".join(sorted(model_names["mode_1"].values()))
    xai_discrete_step_dir = os.path.join(xai_step_dir, "discrete_%s" % model_name)
    util_general.create_dir(xai_discrete_step_dir)
    xai_probs_step_dir = os.path.join(xai_step_dir, "probs_%s" % model_name)
    util_general.create_dir(xai_probs_step_dir)

    # Models
    models = {}
    for model_id, model_name in model_names['mode_1'].items():
        model = torch.load(os.path.join(model_dir_exp, "%s.pt" % model_name), map_location=device)
        model = model.to(device)
        model.eval()
        models[model_id] = model

    for image, label, file_name, raw_image in tqdm(zip(images, labels, file_names, raw_images)):

        region_models = collections.defaultdict(lambda: {})
        for model_id, model_name in model_names['mode_1'].items():
            # Dir
            xai_model_step_dir = os.path.join(xai_step_dir, model_name)

            # Target Layer
            target_layer = util_xai.get_target_layer(model_name)

            # Grad-CAM
            gcam = util_xai.GradCAM(model=models[model_id])
            probs, ids = gcam.forward(image.unsqueeze(0)) # sorted
            #gcam.backward(ids=ids[:, [0]])
            gcam.backward(ids=torch.tensor([[label]], device=device))
            region = gcam.generate(target_layer=target_layer)

            region_models[model_name][file_name] = region

            # Save Grad-CAM
            util_xai.save_gradcam(filename=os.path.join(xai_model_step_dir, "%s.tiff" % file_name),
                                  gcam=region[0][0],
                                  raw_image=raw_image,
                                  paper_cmap=False)

        # GradCam 3 models
        region_discrete = torch.zeros([1, 1, cfg_modes['mode_1']['data']['img_dim'], cfg_modes['mode_1']['data']['img_dim']]).to(device)
        region_probs = torch.zeros([1, 1, cfg_modes['mode_1']['data']['img_dim'], cfg_modes['mode_1']['data']['img_dim']]).to(device)

        # Weighted sum
        for model_name in region_models:
            region_discrete += region_models[model_name][file_name] * weights_discrete[model_name]
            region_probs += region_models[model_name][file_name] * weights_probs["%s_%i" % (model_name, class_labels[0])]

        # Rescale 0:1
        region_discrete = ((region_discrete - region_discrete.min()) / (region_discrete.max() - region_discrete.min()))
        region_probs = ((region_probs - region_probs.min()) / (region_probs.max() - region_probs.min()))

        util_xai.save_gradcam(filename=os.path.join(xai_discrete_step_dir, "%s.tiff" % file_name),
                              gcam=region_discrete[0][0],
                              raw_image=raw_image,
                              paper_cmap=False)
        util_xai.save_gradcam(filename=os.path.join(xai_probs_step_dir, "%s.tiff" % file_name),
                              gcam=region_probs[0][0],
                              raw_image=raw_image,
                              paper_cmap=False)

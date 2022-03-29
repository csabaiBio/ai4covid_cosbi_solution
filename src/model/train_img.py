import sys; print('Python %s on %s' % (sys.version, sys.platform))
import os

filepath = os.path.abspath(__file__)
HOME = '/'.join( filepath.split('/')[:-3] ) + '/'
sys.path.append(HOME)

sys.path.extend(["./"])
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
import collections
import yaml

import src.utils.util_general as util_general
import src.utils.util_data as util_data
import src.utils.util_model as util_model

# Configuration file
args = util_general.get_args()
args.cfg_file = HOME+"configs/img.yaml"
with open(os.path.join(args.cfg_file)) as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

# Seed everything
util_general.seed_all(cfg['seed'])

# Parameters
exp_name = cfg['exp_name']
classes = cfg['data']['classes']
model_list = cfg['model']['model_list']

# Device
device = torch.device(cfg['device']['cuda_device'] if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
print(device)

# Files and Directories
model_dir = os.path.join(cfg['data']['model_dir'], exp_name)
util_general.create_dir(model_dir)

report_dir = os.path.join(cfg['data']['report_dir'], exp_name)
util_general.create_dir(report_dir)
table_dir = os.path.join(report_dir, "tables")
util_general.create_dir(table_dir)
figure_dir = os.path.join(report_dir, "figures")
util_general.create_dir(figure_dir)

report_file = os.path.join(table_dir, 'report.xlsx')
#util_general.delete_file(report_file)

plot_training_dir = os.path.join(figure_dir, "training")
util_general.create_dir(plot_training_dir)

# Train
results = collections.defaultdict(lambda: [])

# Data Loaders
fold_data = {step: pd.read_csv(os.path.join(cfg['data']['fold_dir'], '%s.txt' % step), delimiter=" ", index_col=0) for step in ['train', 'val', 'test']}
#print('FOLD data', fold_data)
datasets = {step: util_data.Dataset( 
                    data=fold_data[step],
                    classes=classes,
                    img_dir=cfg['data']['img_dir'],
                    mask_file=cfg['data']['mask_file'],
                    mask_dir=cfg['data']['mask_dir'],
                    box_file=cfg['data']['box_file'],
                    clahe=cfg['data']['clahe'],
                    step=step,
                    img_dim=cfg['data']['img_dim'])
             for step in ['train', 'val', 'test']}
#print('DATASETS train', dir(datasets['train'].data) )
data_loaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
                'val': torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
                'test': torch.utils.data.DataLoader(datasets['test'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker)}

# Finetuning the convnet
for model_name in model_list:
    # Model
    print("%s%s%s" % ("*"*50, model_name, "*"*50))
    model = util_model.initialize_model(model_name=model_name, num_classes=len(classes), freeze=cfg['model']['freeze'], pretrained=cfg['model']['pretrained'])
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg['trainer']['optimizer']['lr'])
    # LR Scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=cfg['trainer']['scheduler']['mode'], patience=cfg['trainer']['scheduler']['patience'])
    # Train model
    model, history = util_model.train_model(model=model, criterion=criterion, optimizer=optimizer,
                                            scheduler=scheduler, model_name=model_name, data_loaders=data_loaders,
                                            model_dir=model_dir, device=device,
                                            num_epochs=cfg['trainer']['max_epochs'],
                                            early_stopping=cfg['trainer']['early_stopping'])

    # Plot Training
    util_model.plot_training(history=history, model_name=model_name, plot_training_dir=plot_training_dir)

    # Test model
    test_results = util_model.evaluate(model=model, data_loader=data_loaders['test'], device=device)
    print(test_results)

    # Update report
    results["ACC"].append(test_results['all'])
    for c in classes:
        results["ACC %s" % str(c)].append(test_results[c])

    # Save Results
    results_frame = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
    results_frame.insert(loc=0, column='model', value=model_list[:len(results_frame)])
    results_frame.to_excel(report_file, index=False)

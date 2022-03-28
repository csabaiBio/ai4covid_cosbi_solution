import collections

import torch
import torch.nn as nn
import time
import copy
import pandas as pd
from tqdm import tqdm
import os
import math
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np

import src.utils.util_general as util_general


def set_parameter_requires_grad(model, freeze):
    if freeze:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, freeze, pretrained=True):
    if model_name == "alexnet":
        model = models.alexnet(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "vgg11":
        model = models.vgg11(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "vgg11_bn":
        model = models.vgg11_bn(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "vgg13":
        model = models.vgg13(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "vgg13_bn":
        model = models.vgg13_bn(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "vgg16_bn":
        model = models.vgg16_bn(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "vgg19":
        model = models.vgg19(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "vgg19_bn":
        model = models.vgg19_bn(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnext50_32x4d":
        model = models.resnext50_32x4d(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnext101_32x8d":
        model = models.resnext101_32x8d(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "wide_resnet50_2":
        model = models.wide_resnet50_2(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "wide_resnet101_2":
        model = models.wide_resnet101_2(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "squeezenet1_0":
        model = models.squeezenet1_0(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.classifier[1].in_channels
        model.classifier[1] = nn.Conv2d(num_ftrs, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif model_name == "squeezenet1_1":
        model = models.squeezenet1_1(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.classifier[1].in_channels
        model.classifier[1] = nn.Conv2d(num_ftrs, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "densenet169":
        model = models.densenet169(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "densenet161":
        model = models.densenet161(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "densenet201":
        model = models.densenet201(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "inception_v3":
        model = models.inception_v3(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "googlenet":
        model = models.googlenet(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "shufflenet_v2_x0_5":
        model = models.shufflenet_v2_x0_5(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "shufflenet_v2_x1_5":
        model = models.shufflenet_v2_x1_5(pretrained=False)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "shufflenet_v2_x2_0":
        model = models.shufflenet_v2_x2_0(pretrained=False)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "mnasnet0_5":
        model = models.mnasnet0_5(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "mnasnet0_75":
        model = models.mnasnet0_75(pretrained=False)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "mnasnet1_0":
        model = models.mnasnet1_0(pretrained=pretrained)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    elif model_name == "mnasnet1_3":
        model = models.mnasnet1_3(pretrained=False)
        set_parameter_requires_grad(model, freeze)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    else:
        print("Invalid model name, exiting...")
        exit()

    return model


def train_model(model, criterion, optimizer, scheduler, model_name, data_loaders, model_dir, device, num_epochs=25, early_stopping=3):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.Inf

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            with tqdm(total=len(data_loaders[phase].dataset), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='img') as pbar:
                for inputs, labels, file_names in data_loaders[phase]:
                    # print('FFF', file_names)
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs.float())
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        pbar.set_postfix(**{'loss (batch)': loss.item()})

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    pbar.update(inputs.shape[0])

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset)

            if phase == 'val':
                scheduler.step(epoch_loss)

            # update history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_epoch = epoch
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= early_stopping:
                        print(f'\nEarly Stopping! Total epochs: {epoch}%')
                        early_stop = True
                        break

        if early_stop:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:0f}'.format(best_epoch))
    print('Best val Acc: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Save model
    torch.save(model, os.path.join(model_dir, "%s.pt" % model_name))

    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    return model, history


def plot_training(history, model_name, plot_training_dir):
    model_plot_dir = os.path.join(plot_training_dir, model_name)
    util_general.create_dir(model_plot_dir)
    # Training results Loss function
    plt.figure(figsize=(8, 6))
    for c in ['train_loss', 'val_loss']:
        plt.plot(history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Negative Log Likelihood')
    plt.title('Training and Validation Losses')
    plt.savefig(os.path.join(model_plot_dir, "Loss"))
    plt.show()
    # Training results Accuracy
    plt.figure(figsize=(8, 6))
    for c in ['train_acc', 'val_acc']:
        plt.plot(100 * history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.savefig(os.path.join(model_plot_dir, "Acc"))
    plt.show()


def evaluate(model, data_loader, device):
    # Global and Class Accuracy
    correct_pred = {classname: 0 for classname in list(data_loader.dataset.idx_to_class.values()) + ["all"]}
    total_pred = {classname: 0 for classname in list(data_loader.dataset.idx_to_class.values()) + ["all"]}

    # Test loop
    model.eval()
    with torch.no_grad():
        for inputs, labels, file_names in tqdm(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Prediction
            outputs = model(inputs.float())
            _, preds = torch.max(outputs, 1)
            # global
            correct_pred['all'] += (preds == labels).sum().item()
            total_pred['all'] += labels.size(0)
            # class
            for label, prediction in zip(labels, preds):
                if label == prediction:
                    correct_pred[data_loader.dataset.idx_to_class[label.item()]] += 1
                total_pred[data_loader.dataset.idx_to_class[label.item()]] += 1

    # Accuracy
    test_results = {k: correct_pred[k]/total_pred[k] for k in correct_pred.keys() & total_pred}

    return test_results


def predict(model, data_loader, device):
    # Prediction and Truth
    predictions = {}
    probabilities = {}
    truth = {}

    # Predict loop
    model.eval()
    with torch.no_grad():
        for inputs, labels, file_names in tqdm(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Prediction
            outputs = model(inputs.float())
            probs = nn.functional.softmax(outputs, 1)
            _, preds = torch.max(outputs, 1)
            for file_name, label, pred, prob in zip(file_names, labels, preds, probs):
                predictions[file_name] = pred.item()
                probabilities[file_name] = prob.tolist()
                truth[file_name] = label.item()
    return predictions, probabilities, truth


def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


def predict_3model(model, n_models, data_loader, device):
    # Prediction and Truth
    predictions = {}
    probabilities = {}
    truth = {}

    # Predict loop
    model.eval()
    with torch.no_grad():
        for inputs, labels, file_names in tqdm(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Prediction
            if n_models == 2:
                outputs = model(inputs.float(), inputs.float())
            elif n_models == 3:
                outputs = model(inputs.float(), inputs.float(), inputs.float())
            elif n_models == 4:
                outputs = model(inputs.float(), inputs.float(), inputs.float(), inputs.float())
            probs = nn.functional.softmax(outputs, 1)
            _, preds = torch.max(outputs, 1)
            for file_name, label, pred, prob in zip(file_names, labels, preds, probs):
                predictions[file_name] = pred.item()
                probabilities[file_name] = prob.tolist()
                truth[file_name] = label.item()
    return predictions, probabilities, truth


class MLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size0=64, hidden_size1=128, hidden_size2=128, hidden_size3=64, hidden_size4=32, hidden_size5=16, dropout_rate=0.25):
        super(MLP, self).__init__()
        self.fc0 = nn.Sequential(nn.Linear(input_size, hidden_size0), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.fc1 = nn.Sequential(nn.Linear(hidden_size0, hidden_size1), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.fc2 = nn.Sequential(nn.Linear(hidden_size1, hidden_size2), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.fc3 = nn.Sequential(nn.Linear(hidden_size2, hidden_size3), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.fc4 = nn.Sequential(nn.Linear(hidden_size3, hidden_size4), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.fc5 = nn.Sequential(nn.Linear(hidden_size4, hidden_size5), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.fc6 = nn.Sequential(nn.Linear(hidden_size5, num_classes))

    def forward(self, x):
        out = self.fc0(x)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        return out


class MultiLinearMLP(nn.Module):
    def __init__(self, layers):
        super(MultiLinearMLP, self).__init__()
        layer_dict = collections.OrderedDict()
        for i in range(len(layers) - 1):
            layer_dict["layer_%i" % i] = nn.Linear(layers["layer_%i" % i], layers["layer_%i" % (i+1)])
        self.sequential = nn.Sequential(layer_dict)

    def forward(self, x):
        out = self.sequential(x)
        return out

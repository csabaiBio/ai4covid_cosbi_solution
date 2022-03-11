import torch
import torch.nn as nn
import time
import copy
import pandas as pd
from tqdm import tqdm
import os
import numpy as np


def train_multimodal_model(model, n_models, criterion, optimizer, scheduler, model_name, data_loaders, model_dir, device, num_epochs=25, early_stopping=3):
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

            # Iterate over data.
            with tqdm(total=len(data_loaders[phase].dataset), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='img') as pbar:
                for input_1, input_2, labels, file_names in data_loaders[phase]:
                    input_1 = input_1.to(device)
                    input_2 = input_2.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        if n_models == 2:
                            outputs = model(input_1.float(), input_2.float())
                        elif n_models == 3:
                            outputs = model(input_1.float(), input_1.float(), input_2.float())
                        elif n_models == 4:
                            outputs = model(input_1.float(), input_1.float(), input_1.float(), input_2.float())
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        pbar.set_postfix(**{'loss (batch)': loss.item()})

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * input_1.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    pbar.update(input_1.shape[0])

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


def evaluate_multimodal_model(model, n_models, data_loader, device):
    # Global and Class Accuracy
    correct_pred = {classname: 0 for classname in list(data_loader.dataset.idx_to_class.values()) + ["all"]}
    total_pred = {classname: 0 for classname in list(data_loader.dataset.idx_to_class.values()) + ["all"]}

    # Test loop
    model.eval()
    with torch.no_grad():
        for input_1, input_2, labels, file_names in tqdm(data_loader):
            input_1 = input_1.to(device)
            input_2 = input_2.to(device)
            labels = labels.to(device)
            # Prediction
            if n_models == 2:
                outputs = model(input_1.float(), input_2.float())
            elif n_models == 3:
                outputs = model(input_1.float(), input_1.float(), input_2.float())
            elif n_models == 4:
                outputs = model(input_1.float(), input_1.float(), input_1.float(), input_2.float())
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


def predict_multimodal_model(model, n_models, data_loader, device):
    # Prediction and Truth
    predictions = {}
    probabilities = {}
    truth = {}

    # Predict loop
    model.eval()
    with torch.no_grad():
        for input_1, input_2, labels, file_names in tqdm(data_loader):
            input_1 = input_1.to(device)
            input_2 = input_2.to(device)
            labels = labels.to(device)
            # Prediction
            if n_models == 2:
                outputs = model(input_1.float(), input_2.float())
            elif n_models == 3:
                outputs = model(input_1.float(), input_1.float(), input_2.float())
            elif n_models == 4:
                outputs = model(input_1.float(), input_1.float(), input_1.float(), input_2.float())
            probs = nn.functional.softmax(outputs, 1)
            _, preds = torch.max(outputs, 1)
            for file_name, label, pred, prob in zip(file_names, labels, preds, probs):
                predictions[file_name] = pred.item()
                probabilities[file_name] = prob.tolist()
                truth[file_name] = label.item()
    return predictions, probabilities, truth
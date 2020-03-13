import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import random
import pandas as pd
import sys
import os
import copy
from barbar import Bar

def set_parameter_requires_grad(model, feature_extracting):
    """
     
    This helper function sets the ``.requires_grad`` attribute of the
    parameters in the model to False when we are feature extracting. By
    default, when we load a pretrained model all of the parameters have
    ``.requires_grad=True``, which is fine if we are training from scratch
    or finetuning. However, if we are feature extracting and only want to
    compute gradients for the newly initialized layer then we want all of
    the other parameters to not require gradients. This will make more sense
    later.
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
            
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "wide_resnet50":
        """ wide_resnet50_2
        """
        model_ft = models.wide_resnet50_2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "wide_resnet101":
        """ wide_resnet101
        """
        model_ft = models.wide_resnet101_2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnext50":
        """ resnext50_32x4d
        """
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnext101":
        """ resnext101_32x8d
        """
        model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)

    else:
        print("Invalid model name, exiting...")
    return model_ft



def train_complete_model(device, model, dataloaders, criterion, optimizer, num_epochs, save_model_path, is_inception=False):
    """
    The ``train_model`` function handles the training and validation of a
    given model. As input, it takes a PyTorch model, a dictionary of
    dataloaders, a loss function, an optimizer, a specified number of epochs
    to train and validate for, and a boolean flag for when the model is an
    Inception model. The *is_inception* flag is used to accomodate the
    *Inception v3* model, as that architecture uses an auxiliary output and
    the overall model loss respects both the auxiliary output and the final
    output, as described
    `here <https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958>`__.
    The function trains for the specified number of epochs and after each
    epoch runs a full validation step. It also keeps track of the best
    performing model (in terms of validation accuracy), and at the end of
    training returns the best performing model. After each epoch, the
    training and validation accuracies are printed.
    """
    since = time.time()
    training_metrics=np.zeros((num_epochs,4))
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        total_len=0
        # Iterate over data.
        for inputs, labels in Bar(dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                total_len+=len(preds)
                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / total_len
        epoch_acc = running_corrects.double() / total_len

        print('Loss: {:.4f} Acc: {:.4f}'.format( epoch_loss, epoch_acc))

        training_metrics[epoch,[0,1]]=[epoch_loss, epoch_acc]
        torch.save(model.state_dict(), os.path.join(save_model_path + "_"+str(epoch) + '.pt'))
        print()


    fig=plt.figure(figsize=(20, 5))
    fig.add_subplot(1, 2, 1)
    plt.plot(training_metrics[:,[0,2]])
    plt.legend(["training","validation_training"])
    plt.title('Training Loss') 
    fig.add_subplot(1, 2, 2)
    plt.plot(training_metrics[:,[1,3]])
    plt.legend(["training","validation_training"])
    plt.title('Training Accuracy') 
    plt.show()

    return training_metrics


def train_model(device, model, dataloaders, criterion, optimizer, num_epochs, save_model_path, is_inception=False):
    """
    The ``train_model`` function handles the training and validation of a
    given model. As input, it takes a PyTorch model, a dictionary of
    dataloaders, a loss function, an optimizer, a specified number of epochs
    to train and validate for, and a boolean flag for when the model is an
    Inception model. The *is_inception* flag is used to accomodate the
    *Inception v3* model, as that architecture uses an auxiliary output and
    the overall model loss respects both the auxiliary output and the final
    output, as described
    `here <https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958>`__.
    The function trains for the specified number of epochs and after each
    epoch runs a full validation step. It also keeps track of the best
    performing model (in terms of validation accuracy), and at the end of
    training returns the best performing model. After each epoch, the
    training and validation accuracies are printed.
    """
    since = time.time()

    training_metrics=np.zeros((num_epochs,4))
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            total_len=0
            # Iterate over data.
            for inputs, labels in Bar(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    total_len+=len(preds)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / total_len
            epoch_acc = running_corrects.double() / total_len

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch=epoch
            # Save metrics
            if phase == 'train':
                training_metrics[epoch,[0,1]]=[epoch_loss, epoch_acc]
                torch.save(model.state_dict(), os.path.join(save_model_path + "_"+str(epoch) + '.pt'))
            else:
                training_metrics[epoch,[2,3]]=[epoch_loss, epoch_acc]
            # Save model

        print()

    fig=plt.figure(figsize=(20, 5))
    fig.add_subplot(1, 2, 1)
    plt.plot(training_metrics[:,[0,2]])
    plt.legend(["training","validation_training"])
    plt.title('Training Loss') 
    fig.add_subplot(1, 2, 2)
    plt.plot(training_metrics[:,[1,3]])
    plt.legend(["training","validation_training"])
    plt.title('Training Accuracy') 
    plt.show()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, training_metrics,best_epoch




def evaluate_model(device,model, dataloaders, is_inception=False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0


    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    total_len=0
    # Iterate over data.
    for inputs, labels in dataloaders:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.no_grad():
            # Get model outputs and calculate loss
            # Special case for inception because in training it has an auxiliary output. In train
            #   mode we calculate the loss by summing the final output and the auxiliary output
            #   but in testing we only consider the final output.
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total_len+=len(preds)

        # statistics
        running_corrects += torch.sum(preds == labels.data)

    epoch_acc = running_corrects.double() / total_len

    print(' Acc: {:.4f}'.format( epoch_acc))

    print()

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    return epoch_acc.cpu().numpy()

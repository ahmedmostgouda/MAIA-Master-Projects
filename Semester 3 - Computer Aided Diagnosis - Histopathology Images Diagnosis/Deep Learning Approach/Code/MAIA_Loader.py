"""Based on https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
Modified by Maia Team
"""

import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from imgaug import augmenters as iaa
import PIL
from PIL import Image 




def get_train_valid_loader_AUG(data_dir,
                           batch_size,
                           augment,
                           input_size,
                           challange_type,
                           valid_size=0.1,
                           random_seed=8,
                           shuffle=True,
                           num_workers=4):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.Resize(input_size), 
            transforms.ToTensor(),
            normalize,
    ])



  
    if augment:
      if (challenge_type == 1):    #1: "Dermo Challenge"
          train_transform = transforms.Compose([
              transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
              transforms.RandomHorizontalFlip(),
              transforms.RandomVerticalFlip(),
              transforms.RandomRotation(degrees = 90),
              transforms.Resize(input_size),
              transforms.ToTensor(),
              normalize,
          ])
      if (challenge_type == 2):     #2: "Histopathology Challenge"
          train_transform = transforms.Compose([
              transforms.RandomHorizontalFlip(),
              transforms.RandomVerticalFlip(),
              transforms.RandomRotation(degrees = 90),
              transforms.Resize(input_size),
              transforms.ToTensor(),
              normalize,
          ])

    else:
        train_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            normalize,
        ])
    
    # load the dataset
    train_dataset = datasets.ImageFolder(
        root=data_dir,transform=train_transform,
    )

    valid_dataset = datasets.ImageFolder(
        root=data_dir, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers
    )

    dataloaders_dict={"train":train_loader,
                      "val":valid_loader}
    return dataloaders_dict
    
    
    
def get_train_loader_AUG(data_dir,
                           batch_size,
                           augment,
                           input_size,
                           challenge_type,
                           random_seed=8,
                           shuffle=True,
                           num_workers=4):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if augment:
      if (challenge_type == 1):    #1: "Dermo Challenge"
          train_transform = transforms.Compose([
              transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
              transforms.RandomHorizontalFlip(),
              transforms.RandomVerticalFlip(),
              transforms.RandomRotation(degrees = 90),
              transforms.Resize(input_size),
              transforms.ToTensor(),
              normalize,
          ])
      if (challenge_type == 2):     #2: "Histopathology Challenge"
          train_transform = transforms.Compose([
              transforms.RandomHorizontalFlip(),
              transforms.RandomVerticalFlip(),
              transforms.RandomRotation(degrees = 90),
              transforms.Resize(input_size),
              transforms.ToTensor(),
              normalize,
          ])

    else:
        train_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            normalize,
        ])
    
    # load the dataset
    train_dataset = datasets.ImageFolder(
        root=data_dir,transform=train_transform,
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers
    )

    return train_loader


def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           input_size,
                           valid_size=0.1,
                           random_seed=8,
                           shuffle=True,
                           num_workers=4):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.Resize(input_size), 
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = datasets.ImageFolder(
        root=data_dir,transform=train_transform,
    )

    valid_dataset = datasets.ImageFolder(
        root=data_dir, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    print(train_idx, valid_idx)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers
    )

    dataloaders_dict={"train":train_loader,
                      "val":valid_loader}
    return dataloaders_dict
    #return (train_loader, valid_loader)


def get_val_loader(data_dir,
                    batch_size,
                    input_size,
                    shuffle=True,
                    num_workers=4):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.ImageFolder(
        root=data_dir, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers
    )

    return data_loader

def get_test_loader(test_image_path,input_size):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        normalize,
    ])
    image = Image.open(test_image_path)
    
    image = transform(image).float()
    image_expand = torch.tensor(np.expand_dims(image, axis=0))

    data_loader = image_expand.cuda()

    return data_loader
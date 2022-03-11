# Imports
import torch
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import Dataset, random_split

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        
        return tuple_with_path
        
        
# Class that returns dataset in cuda tensors
class ARN_dataset(Dataset):
    '''
    Class transforms loaded dataset into cuda tensors
    Input: data from load_dataset function
    Output: dataset in cuda tensor format
    '''

    def __init__(self, causal, anti):

        # Read csv files and normalize
        c_paths = torch.tensor(StandardScaler().fit_transform(pd.read_csv(causal, header=None)), dtype=torch.float32)
        a_paths = torch.tensor(StandardScaler().fit_transform(pd.read_csv(anti, header=None)), dtype=torch.float32)

        # Transform into tensors
        self.x = torch.unsqueeze(torch.cat((c_paths, a_paths)), dim=1)
        self.y = torch.unsqueeze(torch.cat((torch.zeros(len(c_paths)), torch.ones(len(a_paths)))), dim=1)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.x.shape[0]

def init_dataset(DSET, NUM, DATA, MODEL, CONDITION, train=True):
    """
    :param DSET: Which dataset to use
    :param NUM: Model index
    :param DATA: Whether to use same or different datasets
    :param train: Whether to choose train or test dataset
    :return: Initialized dataset
    """

    # Check which dataset is supposed to be used
    if DSET == "ImageNet":

        # Dataset location
        data_dir = '/INSERT_PATH/'

        # Set normalization parameters
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Check whether train or validation dataset is needed
        if train is True:
            path = os.path.join(data_dir, 'train')

            # Initialize dataset and apply transforms
            dataset = datasets.ImageFolder(
                path,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

        elif train is False:
            path = os.path.join(data_dir, 'val')

            # Initialize dataset and apply transforms
            dataset = datasets.ImageFolder(
                path,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))

    if DSET == "Gaussian":

        # Dataset location
        data_dir = '/INSERT_PATH/'

     
        # Check whether train or validation dataset is needed
        if train is True:
            path = os.path.join(data_dir, 'train')

            # Initialize dataset and apply transforms
            dataset = datasets.ImageFolder(
                path,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]))

        elif train is False:
            path = os.path.join(data_dir, 'val')

            # Initialize dataset and apply transforms
            dataset = datasets.ImageFolder(
                path,
                transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                ]))
   
    if DSET == "CIFAR100":

        # Dataset location
        data_dir = '/INSERT_PATH/'

        # Set normalization parameters
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     
        # Check whether train or validation dataset is needed
        if train is True:
            path = os.path.join(data_dir, 'train')

            # Initialize dataset and apply transforms
            dataset = datasets.ImageFolder(
                path,
                transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]))

        elif train is False:
            path = os.path.join(data_dir, 'val')

            # Initialize dataset and apply transforms
            dataset = datasets.ImageFolder(
                path,
                transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]))
            
    return dataset


def split_dataset(train_set, NUM, GLOBAL_SEED):
    """
    :param train_set: Complete training data set
    :param NUM: Current model index
    :return: Part of training set that belongs to model
    """

    # Length of both datasets should be half of complete dataset
    length = int(len(train_set)/2)

    # Check if remainder of division by 2 is 0, otherwise one sample will be lost
    if (len(train_set) % 2) == 0:
        set_1, set_2 = random_split(train_set, [length, length], generator=torch.Generator().manual_seed(GLOBAL_SEED))
    else:
        set_1, set_2 = random_split(train_set, [length, length+1], generator=torch.Generator().manual_seed(GLOBAL_SEED))

    # Give each set to the respective model
    if NUM == 1:
        train_set = set_1
    elif NUM == 2:
        train_set = set_2
    else:
        raise Exception("Number of models does not match condition.")

    return train_set

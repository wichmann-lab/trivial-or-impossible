# Imports
import torch
import random
import numpy as np
import os


# Function that processes input variables from shell script
def process_sys(sys_vars):
    """
    :param sys_vars: contains input variables from shell script
    :return: returns single values according to variable order
    """

    MODEL = sys_vars[1]  # Model architecture: ConvNet, AlexNet, ResNet
    DSET = sys_vars[2]  # Which dataset to use
    INIT = sys_vars[3]  # Model initialisation: "same" or "different"
    OPTIM = sys_vars[4]  # Optimizer: "SGDN" or "SGD"
    DATA = sys_vars[5]  # Training data: "same" or "different"
    ORDER = sys_vars[6]  # Order of training data: "same" or "different"
    LR = float(sys_vars[7])  # Learning rate
    BATCHES = sys_vars[8]  # Batch size: "same" or "different"
    EPOCHS = int(sys_vars[9])  # Training epochs
    CUDA = int(sys_vars[10])  # CUDA randomness: 0 is deterministic and 1 is random
    NUM = int(sys_vars[11])  # Model index
    VERBOSE = int(sys_vars[12])  # Whether to print training output
    CONDITION = sys_vars[13]  # Name of condition

    # Standard batch size for ResNet18 run
    if (MODEL == "ResNet18" or MODEL == "AlexNet" or MODEL == "ResNet18CIFAR" 
        or MODEL == "DenseNet121CIFAR" or MODEL == "ResNet18FRACTAL" or MODEL == "DenseNet121FRACTAL"):
        BATCH_SIZE = 256

    # Standard batch size for DenseNet121 run, where ResNet50 is different architecture
    if MODEL == "DenseNet121" or MODEL == "VGG11" or MODEL == "ResNet50fc100":
        BATCH_SIZE = 64	

    # For LR condition, models have different LRs: 0.148, 0.149, 0.150, 0.151, 0.152
    if CONDITION.endswith("Different_LR") or CONDITION.endswith("Combined_condition"):
        if MODEL == "ResNet50fc100" or MODEL == "ResNet18fc100":
            LR = LR - 0.00003 + (0.00001 * NUM)
        else:
            LR = LR - 0.003 + (0.001 * NUM)
            
    return MODEL, DSET, INIT, OPTIM, DATA, ORDER, LR, BATCHES, EPOCHS, CUDA, NUM, VERBOSE, CONDITION, BATCH_SIZE


# Function that prints parameters for manual check
def parameter_check(MODEL, DSET, INIT, OPTIM, DATA, ORDER, LR, BATCHES, EPOCHS, CUDA, NUM, VERBOSE, DEVICE, CONDITION):
    """
    :param MODEL: Model architecture: ConvNet, AlexNet, ResNet, str
    :param DSET: Which dataset to use, str
    :param INIT: Model initialisation: "same" or "different", str
    :param OPTIM: Optimizer: "SGDN" or "SGD", str
    :param DATA: Training data: "same" or "different", str
    :param ORDER: Order of training data: "same" or "different", str
    :param LR: Learning rate, float
    :param BATCHES: Batch size: "same" or "different", str
    :param EPOCHS: Training epochs, int
    :param CUDA: CUDA randomness: 0 is deterministic and 1 is random, int
    :param NUM: Model index, int
    :param VERBOSE: Whether to print training output, int
    :param DEVICE: Which device is used: CUDA or cpu, str
    :param CONDITION: Name of condition, str
    :return: print string containing each parameter and it's assigned value
    """
    print("_____Condition: {}_____".format(CONDITION))
    print("Model architecture: {}".format(MODEL))
    print("Dataset: {}".format(DSET))
    print("Initialisation: {}".format(INIT))
    print("Optimizer: {}".format(OPTIM))
    print("Training data: {}".format(DATA))
    print("Order of training data: {}".format(ORDER))
    print("Learning rate: {}".format(LR))
    print("Batch condition: {}".format(BATCHES))
    print("Number of epochs: {}".format(EPOCHS))
    print("CUDA randomness: {}".format(CUDA))
    print("Model index: {}".format(NUM))
    print("Print training output: {}".format(VERBOSE))
    print("Using device: {}".format(DEVICE))

    return


# Function that checks whether this condition has already been run
def run_check(filename, EPOCHS):
    """
    :param filename: Filename to check folder
    :return: Nothing, exception called when condition has already been run
    """
    # Check how many data entries there are for this model, subtract 4 arrays which are stored in the end
    length = (len(os.listdir(filename)) - 4) / 2

    # Two files each epoch (thus len / 2), and EPOCHS + 1 in total
    if length == (EPOCHS + 1):
        raise Exception("Condition has already been run!")

    return


# Function that sets global seeds
def set_seeds(SEED):
    """
    :param SEED: Seed to set
    :return: Seeds set for numpy, python random and torch
    """
    np.random.seed(SEED)  # Numpy seed
    random.seed(SEED)  # Python random seed
    torch.manual_seed(SEED)  # Torch seed

    return


# Function that sets global seeds
def set_cuda_randomness(CUDA):
    """
    :param CUDA: CUDA randomness: 0 is deterministic and 1 is random, int
    :return: Set cudnn to deterministic or not
    """
    # Check CUDA condition
    if CUDA == 1:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    elif CUDA == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        raise Exception("No CUDA randomness strategy defined!")

    return


# Function that sets optimizer
def set_optim(OPTIM, model, LR):
    """
    :param OPTIM: Name of optimizer to use
    :return: Return optimizer
    """

    if OPTIM == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    elif OPTIM == "SGDN":
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4, nesterov=True)
    elif OPTIM == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    elif OPTIM == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=LR)
    elif OPTIM == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LR)
    elif OPTIM == "Rprop":
        optimizer = torch.optim.Rprop(model.parameters(), lr=LR)
    elif OPTIM == "ASGD":
        optimizer = torch.optim.ASGD(model.parameters(), lr=LR)
    else:
        raise Exception("No optimizer defined!")

    return optimizer


# Function that return the filename for the output files
def make_info(MODEL, DSET, INIT, OPTIM, DATA, ORDER, LR, BATCHES, EPOCHS, CUDA, NUM, CONDITION):
    """
    :param MODEL: Model architecture: ConvNet, AlexNet, ResNet, str
    :param DSET: Which dataset to use, str
    :param INIT: Model initialisation: "same" or "different", str
    :param OPTIM: Optimizer: "Adam" or "SGD", str
    :param DATA: Training data: "same" or "different", str
    :param ORDER: Order of training data: "same" or "different", str
    :param LR: Learning rate, float
    :param BATCHES: Batch size: "same" or "different", str
    :param EPOCHS: Training epochs, int
    :param CUDA: CUDA randomness: 0 is deterministic and 1 is random, int
    :param NUM: Model index, int
    :return: Return filename string and information array
    """

    # Define path name
    filename = "./results/{}/NUM{}/".format(CONDITION, NUM)

    # Create path if it does not exist
    if not os.path.exists(filename):
        os.makedirs(filename)

    # Create info array
    info = [["MODEL", MODEL], ["DSET", DSET], ["INIT", INIT], ["OPTIM", OPTIM], ["DATA", DATA], ["ORDER", ORDER],
            ["LR", LR], ["BATCHES", BATCHES], ["EPOCHS", EPOCHS], ["CUDA", CUDA], ["NUM", NUM]]

    return filename, info


def lr_update(epoch, LR, EPOCHS):
    """
    :param epoch: Which epoch are we in?
    :param LR: Starting learning rate
    :return: Learning rate for current epoch
    """

    # LR scheduler for ResNet18 run: model uses 90 epochs in total
    if EPOCHS == 90 or EPOCHS == 100:

        # Change learning rate depending on epoch number
        lrTmp = LR

        if (epoch > 29):
            lrTmp = LR / 10

        if (epoch > 59):
            lrTmp = LR / 100

    # For DenseNet121 run, different LR scheduler is used: model only uses 45 epochs in total
    elif EPOCHS == 30 or EPOCHS == 40:

        # Change learning rate depending on epoch number
        lrTmp = LR

        if (epoch > 9):
            lrTmp = LR / 10

        if (epoch > 19):
            lrTmp = LR / 100

    # If number of epochs does not match, raise an exception
    else:

        raise Exception("No suitable number of epochs defined!")

    return lrTmp


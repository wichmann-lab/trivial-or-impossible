# Imports
import torch
from torchvision.models import resnet18, resnet50, densenet121, vgg11, alexnet
from scripts.models import ConvNet, AltConvNet
import torch.nn as nn

# Function that initializes model
def init_model(INIT, INIT_SEED, MODEL, NUM, DEVICE):
    """
    :param INIT: Whether to initialize models "same" or "different"
    :param INIT_SEED: Seed for initialization
    :param MODEL: Model architecture
    :param NUM: Model index
    :param DEVICE: Device on which model should run
    :return: Initialized model
    """
    # Check model architecture condition
    if MODEL == "ResNet18":
        architecture = resnet18
    elif MODEL == "ResNet50":
        architecture = resnet50
    elif MODEL == "ResNet18fc100":
        architecture = resnet18
    elif MODEL == "ResNet18CIFAR":
        architecture = resnet18
    elif MODEL == "ResNet18FRACTAL":
        architecture = resnet18
    elif MODEL == "ResNet50fc100":
        architecture = resnet50
    elif MODEL == "DenseNet121":
        architecture = densenet121
    elif MODEL == "DenseNet121CIFAR":
        architecture = densenet121
    elif MODEL == "DenseNet121FRACTAL":
        architecture = densenet121
    elif MODEL == "VGG11":
        architecture = vgg11
    elif MODEL == "AlexNet":
        architecture = alexnet
    elif MODEL == "ConvNet":
        architecture = ConvNet
    elif MODEL == "AltConvNet":
        architecture = AltConvNet
    else:
        raise Exception("No model defined!")

    # Check init condition
    if INIT == "same":
        torch.manual_seed(INIT_SEED)  # Set INIT_SEED to be sure
        model = architecture(pretrained=False)
    elif INIT == "different":
        torch.manual_seed(INIT_SEED + NUM)  # Seed is different for every model
        model = architecture(pretrained=False)
    else:
        raise Exception("No initialization defined!")

    # Change models for dataset other than ImageNet
    if  MODEL == "ResNet18fc100":
        model.fc = nn.Linear(512, 100)
    if  MODEL == "ResNet50fc100":
        model.fc = nn.Linear(2048, 100)
    if  MODEL == "ResNet18CIFAR":
        model.fc = nn.Linear(512, 100)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    if  MODEL == "DenseNet121CIFAR":
        model.classifier = nn.Linear(in_features=1024, out_features=100, bias=True)
        model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Put model on GPU
    model = model.to(DEVICE)
    
    return model


# Function that checks whether initialization worked
def init_check(model, INIT, INIT_SEED, MODEL, DEVICE):
    """
    :param model: Initialized model
    :param INIT: Initialization condition (same or different)
    :param DEVICE: Device on which model should run
    :return: Output string returning whether initialization worked
    """

    # Initialize base model
    base_model = init_model("same", INIT_SEED, MODEL, 0, DEVICE)

    # Get parameter weights for base model and current model
    base_weights = []
    model_weights = []

    for p in base_model.parameters():
        base_weights.append(p)

    for p in model.parameters():
        model_weights.append(p)

    # Calculate difference between the two lists
    difference = 0
    for ind in range(len(base_weights)):
        difference += torch.sum(base_weights[ind] - model_weights[ind])

    # Check INIT condition
    if INIT == "different" and difference != 0:
        print("Success, weights are different.")
    elif INIT == "same" and difference == 0:
        print("Success, weights are the same.")
    else:
        raise Exception("Initialization error!")
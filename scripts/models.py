# Imports
import torch


# Define actual network class
class ConvNet(torch.nn.Module):
    '''
    Class that defines a very simple convolutional network.
    '''

    # Single layer network
    def __init__(self):
        super(ConvNet, self).__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=12, kernel_size=11, stride=1, padding=1),
            torch.nn.ReLU(),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1104, 1),
            torch.nn.Sigmoid(),
        )

    # Define forward pass
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class AltConvNet(torch.nn.Module):
    '''
    Class that defines an alternative very simple convolutional network.
    '''

    # Single layer network
    def __init__(self):
        super(AltConvNet, self).__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=6, kernel_size=11, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=6, out_channels=12, kernel_size=11, stride=1, padding=1),
            torch.nn.ReLU(),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1008, 1),
            torch.nn.Sigmoid(),
        )

    # Define forward pass
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

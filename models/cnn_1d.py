import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from libraries import *

#######################################################################################
#######################################################################################
#######################################################################################

class JustoLiuNet1D_torch(nn.Module):
    """
    JustoLiuNet1D_torch is a 1D Convolutional Neural Network (CNN) model for hyperspectral data classification.
    It consists of several convolutional layers followed by max pooling and a fully connected layer for classification.
    The model is designed to process 1D data, such as spectral data from hyperspectral images.
    The architecture includes:
    - Convolutional layers with ReLU activation functions.
    - Max pooling layers to reduce the spatial dimensions.
    - A fully connected layer for classification.
    The model can be initialized with a specified number of input features, number of classes, kernel size, and starting number of kernels.
    """
    def __init__(self, num_features, num_classes=3, kernel_size=6, starting_kernels=6):
        """
        Initializes the JustoLiuNet1D_torch model.
        Args:
            num_features (int): The number of input features (bands) in the hyperspectral data.
            num_classes (int, optional): The number of output classes for classification. Defaults to 3.
            kernel_size (int, optional): The size of the convolutional kernels. Defaults to 6.
            starting_kernels (int, optional): The number of kernels in the first convolutional layer. Defaults to 6.
        """
        super(JustoLiuNet1D_torch, self).__init__()

        self.conv1 = nn.Conv1d(1, starting_kernels, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(starting_kernels, starting_kernels * 2, kernel_size=kernel_size)
        self.conv3 = nn.Conv1d(starting_kernels * 2, starting_kernels * 3, kernel_size=kernel_size)
        self.conv4 = nn.Conv1d(starting_kernels * 3, starting_kernels * 4, kernel_size=kernel_size)

        self.classifier = nn.Linear(48, num_classes) 

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)

        #print("DEBUG - Shape before flatten:", x.shape)

        x = x.view(x.size(0), -1)

        if self.classifier is None:
            raise RuntimeError(f"'classifier' is not defined yet. Set Linear input dim to {x.size(1)} in __init__.")

        return self.classifier(x)
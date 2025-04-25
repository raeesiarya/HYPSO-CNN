import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from libraries import *

#######################################################################################
#######################################################################################
#######################################################################################

class JustoLiuNet1D_torch(nn.Module):
    def __init__(self, num_features, num_classes=3, kernel_size=6, starting_kernels=6):
        super(JustoLiuNet1D_torch, self).__init__()

        self.conv1 = nn.Conv1d(1, starting_kernels, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(starting_kernels, starting_kernels * 2, kernel_size=kernel_size)
        self.conv3 = nn.Conv1d(starting_kernels * 2, starting_kernels * 3, kernel_size=kernel_size)
        self.conv4 = nn.Conv1d(starting_kernels * 3, starting_kernels * 4, kernel_size=kernel_size)

        self.classifier = nn.Linear(48, num_classes) 

    def forward(self, x):
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
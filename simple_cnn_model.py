import torch.nn as nn

class SimpleMNIST_CNN(nn.Module):

    def __init__(self):
        super().__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten layer (no parameters, just reshaping)
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(64 * 7 * 7, 10)  # Adjusted for MNIST input size

    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Second convolutional block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten the output for the fully connected layer  
        x = self.flatten(x)

        # Fully connected layer
        x = self.fc(x)
        return x



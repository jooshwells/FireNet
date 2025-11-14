import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()

        # Define various layers here, such as in the tutorial example

        ############### PART 1 #################
        # >2 conv layers and >1 fc layers
        # CIFAR10 image = 32x32; 3 input channels (RGB)

        # convolutional layers
        self.conv1 = nn.Conv2d(3, 40, 3, 1, 1) # 32x32x40
        self.conv2 = nn.Conv2d(40, 80, 3, 1, 1) # 16x16x80
        self.conv3 = nn.Conv2d(80, 160, 3, 1, 1) # 8x8x160

        # pooling
        self.pool = nn.MaxPool2d(2, 2) # halves dimensions

        # fully connected layers
        self.fc1 = nn.Linear(4 * 4 * 160, 400) # size after conv layers = 4x4x160
        self.fc2 = nn.Linear(400, 400)

        # output layer
        self.fcOut = nn.Linear(400, 10) # 10 output for the 10 CIFAR10 classes


        ############### PART 2 #################
        # more convolutional layers added

        # convolutional layers
        self.conv4 = nn.Conv2d(160, 80, 3, 1, 1) # 4x4x80
        self.conv5 = nn.Conv2d(80, 160, 3, 1, 1) # 2x2x160

        # fully connected layers
        self.fc3 = nn.Linear(160, 400)
        self.fc4 = nn.Linear(400, 400)

        ########### MODE SELECTION #############
        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for part 1 and 2.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        else:
            print("Invalid mode ", mode, "selected. Select 1 or 2")
            exit(0)

    # Baseline model. Part 1
    def model_1(self, X):
        # ======================================================================
        # More than two conv layers followed by more than 1 fully connected layer.
        # ----------------- YOUR CODE HERE ----------------------

        # send through convolutional layers
        x = self.conv1(X)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        # linearize in order to pass through nn.Linear()
        x = x.view(-1, 4 * 4 * 160) #size after convolutional layers

        # fully connected layers
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        # send through output layer
        x = self.fcOut(x)

        return x

    # More convolutional layers. Part 2
    def model_2(self, X):
        # ======================================================================
        # More convolutional layers than Model 1.
        # ----------------- YOUR CODE HERE ----------------------

        # send through extra convolutional layers
        x = self.conv1(X)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool(x)

        # linearize in order to pass through nn.Linear()
        x = x.view(-1, 160)

        # fully connected layers
        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.relu(x)

        # send through output layer
        x = self.fcOut(x)

        return x


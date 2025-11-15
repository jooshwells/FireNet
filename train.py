from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from ConvNet import ConvNet
import argparse
import numpy as np
import matplotlib.pyplot as plt
import fiftyone as fo
import fiftyone.utils.random as four
from PIL import Image
from FirearmsDataset import FirearmsDataset


def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''

    # Set model to train mode before each epoch
    model.train()

    # Empty list to store losses
    losses = []
    correct = 0

    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample

        # Push data/label to correct device
        data, target = data.to(device), target.to(device)

        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()

        # Do forward pass for current set of data
        output = model(data)

        # ======================================================================
        # Compute loss based on criterion
        # ----------------- YOUR CODE HERE ----------------------
        loss = criterion(output, target)

        # Computes gradient based on final loss
        loss.backward()

        # Store loss
        losses.append(loss.item())

        # Optimize model parameters based on learning rate and gradient
        optimizer.step()

        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=True)

        # ======================================================================
        # Count correct predictions overall
        # ----------------- YOUR CODE HERE ----------------------
        numCorrPred = pred.eq(target.view_as(pred))
        correct += numCorrPred.sum().item()

    train_loss = float(np.mean(losses))
    train_acc = correct / ((batch_idx + 1) * batch_size)
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        float(np.mean(losses)), correct, (batch_idx + 1) * batch_size,
                                         100. * correct / ((batch_idx + 1) * batch_size)))
    return train_loss, train_acc


def test(model, device, test_loader, criterion):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''

    # Set model to eval mode to notify all layers.
    model.eval()

    losses = []
    correct = 0

    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)

            # Predict for data by doing forward pass
            output = model(data)

            # ======================================================================
            # Compute loss based on same criterion as training
            # ----------------- YOUR CODE HERE ----------------------
            # Compute loss based on same criterion as training
            loss = criterion(output, target)

            # Append loss to overall test loss
            losses.append(loss.item())

            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=True)

            # ======================================================================
            # Count correct predictions overall
            # ----------------- YOUR CODE HERE ----------------------
            numCorrPred = pred.eq(target.view_as(pred))
            correct += numCorrPred.sum().item()

    test_loss = float(np.mean(losses))
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return test_loss, accuracy


def run_main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set proper device based on cuda availability
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)

    # Initialize the model and send to device
    model = ConvNet(FLAGS.mode).to(device)

    # ======================================================================
    # Define loss function.
    # ----------------- YOUR CODE HERE ----------------------
    criterion = nn.CrossEntropyLoss()

    # ======================================================================
    # Define optimizer function.
    # ----------------- YOUR CODE HERE ----------------------
    optimizer = optim.SGD(model.parameters(), lr=0.03)

    # Create transformations to apply to each data sample
    # Can specify variations such as image flip, color flip, random crop, ...
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load dataset
    dataset_name = "firearm-detection-final"
    dataset = fo.load_dataset(dataset_name)

    # Split dataset into train and test
    four.random_split(dataset, {"train": 0.8, "test": 0.2})
    train_view = dataset.match_tags("train")
    test_view = dataset.match_tags("test")

    print("Train samples:", len(train_view))
    print("Test samples:", len(test_view))

    # Convert sets to PyTorch
    train_dataset = FirearmsDataset(train_view, transform=transform)
    test_dataset = FirearmsDataset(test_view, transform=transform)

    # Make loaders
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False)

    best_accuracy = 0.0

    # Track loss/acc data for plotting
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    # Run training for n_epochs specified in config
    for epoch in range(1, FLAGS.num_epochs + 1):
        train_loss, train_accuracy = train(model, device, train_loader,
                                           optimizer, criterion, epoch, FLAGS.batch_size)
        test_loss, test_accuracy = test(model, device, test_loader, criterion)

        # track train and test data for plots
        train_loss_list.append(train_loss)
        train_acc_list.append(train_accuracy)

        test_loss_list.append(test_loss)
        test_acc_list.append(test_accuracy)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

    print("accuracy is {:2.2f}\n".format(best_accuracy))

    print("Loss and Accuracy Plots Being Saved")

    # Make training accuracy into percentage
    for x in range(FLAGS.num_epochs):
        train_acc_list[x] *= 100.0

    epochs = range(1, FLAGS.num_epochs + 1)  # Epoch number

    # Setup 2x2 plot and plot training and testing accuracy
    figure, axis = plt.subplots(2, 2, figsize=(12, 10))

    axis[0, 0].plot(epochs, train_loss_list)
    axis[0, 0].set_title('Training Loss')
    axis[0, 0].set_xlabel('Epochs')
    axis[0, 0].set_ylabel('Loss')

    axis[0, 1].plot(epochs, train_acc_list)
    axis[0, 1].set_title('Training Accuracy')
    axis[0, 1].set_xlabel('Epochs')
    axis[0, 1].set_ylabel('Accuracy (%)')

    axis[1, 0].plot(epochs, test_loss_list)
    axis[1, 0].set_title('Test Loss')
    axis[1, 0].set_xlabel('Epochs')
    axis[1, 0].set_ylabel('Loss')

    axis[1, 1].plot(epochs, test_acc_list)
    axis[1, 1].set_title('Test Accuracy')
    axis[1, 1].set_xlabel('Epochs')
    axis[1, 1].set_ylabel('Accuracy (%)')

    plt.suptitle(f'Model {FLAGS.mode}')
    plt.savefig(f'Model{FLAGS.mode}TrainTestPlots.png')
    plt.show()

    print("Training and evaluation finished")


if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-5.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=60,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=32,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')

    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()

    run_main(FLAGS)


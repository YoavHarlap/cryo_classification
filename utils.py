import torch, torchvision
from torch import nn
from torch import optim
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt

import requests
from PIL import Image
from io import BytesIO

import copy

# from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import torch.nn as nn


def create_lenet(dropout_prob =0.5):
    MNIST = False
    if MNIST:
        model = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
    else:
        # model = nn.Sequential(
        #     nn.Conv2d(1, 16, 3, padding=1),  # Increase the number of output channels
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, 3, padding=1),  # Additional convolutional layer
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #
        #     nn.Conv2d(32, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, 3, padding=1),  # Additional convolutional layer
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #
        #     nn.Conv2d(64, 128, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, 3, padding=1),  # Additional convolutional layer
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #
        #     nn.Flatten(),
        #
        #     nn.Linear(128 * 32 * 32, 512),  # Adjust the input size for flattened features
        #     nn.ReLU(),
        #     nn.Linear(512, 256),  # Larger fully connected layers
        #     nn.ReLU(),
        #     nn.Linear(256, 2)
        # )
        # model = nn.Sequential(
        #     nn.Conv2d(1, 16, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #
        #     nn.Conv2d(16, 32, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #
        #     nn.Flatten(),
        #
        #     nn.Linear(32 * 64 * 64, 128),  # Adjust input size
        #     nn.ReLU(),
        #     nn.Linear(128, 2)
        # print
        # )

        # model = nn.Sequential(
        #     nn.Conv2d(1, 6, 5, padding=2),
        #     nn.ReLU(),
        #     nn.AvgPool2d(2, stride=2),
        #     nn.Conv2d(6, 16, 5, padding=0),
        #     nn.ReLU(),
        #     nn.AvgPool2d(2, stride=2),
        #     nn.Flatten(),
        #     nn.Linear(16 * 61 * 61, 120),  # Adjust input size based on feature map size after convolutions
        #     nn.ReLU(),
        #     nn.Linear(120, 84),
        #     nn.ReLU(),
        #     nn.Linear(84, 2)  # Output 2 classes
        # )
        #
        # model = nn.Sequential(
        #     nn.Conv2d(1, 6, 5, padding=2),
        #     nn.ReLU(),
        #     nn.AvgPool2d(2, stride=2),
        #     nn.Conv2d(6, 16, 5, padding=0),
        #     nn.ReLU(),
        #     nn.AvgPool2d(2, stride=2),
        #     nn.Flatten(),
        #     nn.Linear(16 * 32 * 32, 120),  # Correct input size based on feature map size after convolutions
        #     nn.ReLU(),
        #     nn.Linear(120, 84),
        #     nn.ReLU(),
        #     nn.Linear(84, 2)  # Output 2 classes
        # # )
        # workkkk


        model = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 62 * 62, 120),  # Correct input size based on feature map size after convolutions
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2)  # Output 2 classes
        )

        # model = nn.Sequential(
        #     nn.Conv2d(1, 6, 5, padding=2),
        #     nn.ReLU(),
        #     nn.AvgPool2d(2, stride=2),
        #     nn.Conv2d(6, 16, 5, padding=0),
        #     nn.ReLU(),
        #     nn.AvgPool2d(2, stride=2),
        #     nn.Flatten(),
        #     nn.Linear(16 * 62 * 62, 120),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout_prob),  # Add dropout after the first fully connected layer
        #     nn.Linear(120, 84),
        #     nn.ReLU(),
        #     nn.Linear(84, 2)
        # )


    return model


def validate(model, data, device="cpu"):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(data):
        images = images.to(device)
        x = model(images)
        value, pred = torch.max(x, 1)
        pred = pred.data.cpu()
        total += x.size(0)
        correct += torch.sum(pred == labels)
    return correct * 100. / total


def train(train_dl, val_dl, numb_epoch=3, lr=1e-3, device="cpu"):
    accuracies = []
    test_accuracies = []
    loss_array = []
    cnn = create_lenet().to(device)
    cec = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=lr)
    max_accuracy = 0
    prev_accuracy = 100
    counter = 0
    for epoch in range(numb_epoch):
        for i, (images, labels) in enumerate(train_dl):
            images = images.to(device)

            # hanfatza
            labels = labels.type(torch.LongTensor)  # casting to long

            labels = labels.to(device)
            optimizer.zero_grad()
            pred = cnn(images)
            loss = cec(pred, labels)
            loss_array.append(loss)
            loss.backward()

            # Check gradients for NaN or Inf
            has_nan = any(torch.isnan(param.grad).any() for param in cnn.parameters())
            has_inf = any(torch.isinf(param.grad).any() for param in cnn.parameters())
            if has_nan or has_inf:
                print("Gradients contain NaN or Inf values!")
            # else:
            #     print("Gradients is OK!")



            optimizer.step()
        # accuracy = float(validate(cnn, val_dl, device=device))
        accuracy = float(validate(cnn, train_dl, device=device))
        test_acc = float(validate(cnn, val_dl, device=device))

        test_accuracies.append(test_acc)
        accuracies.append(accuracy)
        if accuracy > max_accuracy:
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy
            print("Saving Best Model with Accuracy: ", accuracy)
        # if accuracy <= prev_accuracy:
        #     counter = counter + 1
        # if counter == 3:
        #     break
        print('Epoch:', epoch + 1, "/", numb_epoch, "Accuracy :", accuracy, '%',"loss:",loss.item(),"test_acc:",test_acc)

        prev_accuracy = accuracy
    # plt.plot(accuracies)
    # return best_model,accuracy


    return loss_array,accuracy


def predict_dl(model, data, device="cpu"):
    y_pred = []
    y_true = []
    for i, (images, labels) in enumerate(data):
        images = images.to(device)
        x = model(images)
        value, pred = torch.max(x, 1)
        pred = pred.data.cpu()
        y_pred.extend(list(pred.numpy()))
        y_true.extend(list(labels.numpy()))
    return np.array(y_pred), np.array(y_true)


def inference(path, model, device):
    T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    r = requests.get(path)
    with BytesIO(r.content) as f:
        img = Image.open(f).convert(mode="L")
        img = img.resize((28, 28))
        x = (255 - np.expand_dims(np.array(img), -1)) / 255.
        # from scipy.ndimage import gaussian_filter
        # x = gaussian_filter(x, sigma=3)

    with torch.no_grad():
        pred = model(torch.unsqueeze(T(x), axis=0).float().to(device))
        return F.softmax(pred, dim=-1).cpu().numpy()

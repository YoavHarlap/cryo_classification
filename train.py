import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from model import Model  # Make sure you have a file named 'model.py' with the Model class defined


def plot_accuracy(epochs, accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, accuracies, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.grid(True)
    plt.savefig('accuracy_plot.png')
    plt.show()

def show_misclassified_images(test_loader, model, device):
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    correct_images = []
    correct_labels = []

    for idx, (test_x, test_label) in enumerate(test_loader):
        test_x = test_x.to(device)
        test_label = test_label.to(device)
        predict_y = model(test_x.float()).detach()
        predict_y = torch.argmax(predict_y, dim=-1)

        misclassified_indices = (predict_y != test_label).nonzero().squeeze()
        correct_indices = (predict_y == test_label).nonzero().squeeze()
        print("Shape of predict_y:", predict_y.shape)
        print("Shape of misclassified_indices:", misclassified_indices.shape)

        if predict_y.shape[0] == 0:
            print("Predictions array is empty.")
        else:
            if misclassified_indices.shape[0] == 0:
                print("Misclassified indices array is empty.")
            else:
                print("Attempting indexing...")
                misclassified_labels.extend(predict_y[misclassified_indices].cpu().numpy())
                print("Indexing successful.")

        misclassified_images.extend(test_x[misclassified_indices])
        misclassified_labels.extend(predict_y[misclassified_indices].cpu().numpy())  # Convert to NumPy array
        correct_images.extend(test_x[correct_indices])
        correct_labels.extend(predict_y[correct_indices].cpu().numpy())  # Convert to NumPy array

        if len(misclassified_images) >= 5 and len(correct_images) >= 5:
            break

    misclassified_images = torch.stack(misclassified_images)
    correct_images = torch.stack(correct_images)

    plt.figure(figsize=(15, 8))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(misclassified_images[i].cpu().numpy().squeeze(), cmap='gray')
        plt.title(f"Misclassified\nPred: {misclassified_labels[i]}, Correct: {test_label[misclassified_indices][i].item()}")
        plt.axis('off')

        plt.subplot(2, 5, i + 6)
        plt.imshow(correct_images[i].cpu().numpy().squeeze(), cmap='gray')
        plt.title(f"Correctly Classified\nPred: {correct_labels[i]}, Correct: {test_label[correct_indices][i].item()}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('misclassified_images.png')
    plt.show()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 256
    # Load the MNIST dataset
    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor(), download=True)
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor(), download=True)

    # Filter dataset to only include classes 0 and 1
    train_indices = (train_dataset.targets == 0) | (train_dataset.targets == 1)
    test_indices = (test_dataset.targets == 0) | (test_dataset.targets == 1)

    train_dataset = torch.utils.data.Subset(train_dataset, np.where(train_indices)[0])
    test_dataset = torch.utils.data.Subset(test_dataset, np.where(test_indices)[0])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = Model().to(device)
    sgd = SGD(model.parameters(), lr=1e-1)
    loss_fn = CrossEntropyLoss()
    all_epoch = 2
    prev_acc = 0
    accuracies = []
    epochs = []

    for current_epoch in range(all_epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            sgd.zero_grad()
            predict_y = model(train_x.float())
            loss = loss_fn(predict_y, train_label.long())
            loss.backward()
            sgd.step()

        all_correct_num = 0
        all_sample_num = 0
        model.eval()

        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            predict_y = model(test_x.float()).detach()
            predict_y = torch.argmax(predict_y, dim=-1)
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc = all_correct_num / all_sample_num
        accuracies.append(acc)
        epochs.append(current_epoch)
        print('Epoch {}: accuracy: {:.3f}'.format(current_epoch, acc), flush=True)

        if not os.path.isdir("models"):
            os.mkdir("models")
        torch.save(model.state_dict(), 'models/binary_mnist_{:.3f}.pth'.format(acc))

        if np.abs(acc - prev_acc) < 1e-4:
            break
        prev_acc = acc

    plot_accuracy(epochs, accuracies)
    print("Model finished training")

    # Show misclassified and correctly classified images
    show_misclassified_images(test_loader, model, device)

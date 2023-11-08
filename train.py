import matplotlib.pyplot as plt
import sys

log_file_path = "save_logs.txt"
log_file_path = "/home/yoavharlap/PycharmProjects/cryo_classification/save_logs.txt"
print("yoav")
# Custom file-like object that writes to both stdout and a file
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

# Create a log file to write to
log_file = open(log_file_path, "w")

# Redirect sys.stdout to the custom Tee object
sys.stdout = Tee(sys.stdout, log_file)

# Rest of your code...

# Now, all print statements will write to both console and the log file

#print("This message will be logged to the file and printed to the console.")

from utils import *
from data import cryo_np_Dataset
from torch.utils.data import DataLoader

numb_batch = 64
MNIST = True
MNIST = False
print("MNIST:", MNIST)

if MNIST:
    T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_data = torchvision.datasets.MNIST('mnist_data', train=True, download=True, transform=T)
    val_data = torchvision.datasets.MNIST('mnist_data', train=False, download=True, transform=T)

    train_dl = torch.utils.data.DataLoader(train_data, batch_size=numb_batch)
    val_dl = torch.utils.data.DataLoader(val_data, batch_size=numb_batch)

else:
    # Transforms for custom images (you can customize this)
    custom_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    # Paths to your numpy files
    outliers_file_path = "/data/yoavharlap/10028_classification/outliers_images.npy"
    particles_file_path = "/data/yoavharlap/10028_classification/particles_images.npy"

    outliers_data = np.load(outliers_file_path)
    particles_data = np.load(particles_file_path)
    data = np.concatenate((outliers_data, particles_data), axis=0)
    labels = np.concatenate((np.ones(len(outliers_data)), np.zeros(len(particles_data))))
    train_ratio = 0.8
    total_samples = len(labels)
    train_samples = int(train_ratio * total_samples)

    # Create an index array to shuffle data and labels in the same way
    shuffle_indices = np.arange(len(data))
    np.random.shuffle(shuffle_indices)

    # Shuffle data and labels using the shuffled indices
    shuffled_data = data[shuffle_indices]
    shuffled_labels = labels[shuffle_indices]

    # Create custom datasets for train and validation
    train_data = cryo_np_Dataset(shuffled_data[:train_samples], shuffled_labels[:train_samples], train=True,
                                 transform=custom_transforms)
    val_data = cryo_np_Dataset(shuffled_data[train_samples:], shuffled_labels[train_samples:], train=False,
                               transform=custom_transforms)

    #
    # ##########
    # train_data = train_data[:50]
    # val_data = val_data[:50]
    # ##########

    # Create dataloaders
    train_dl = DataLoader(train_data, batch_size=numb_batch, shuffle=False)
    val_dl = DataLoader(val_data, batch_size=numb_batch, shuffle=False)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Cuda Available!!!!!!!!")
else:
    device = torch.device("cpu")
    print("No Cuda Available")

# List of parameter options to try
# learning_rates = [1e-5, 1e-6, 1e-7]
# num_epochs_list = [40, 60, 80]
learning_rates = [1e-4,1e-5,1e-6]
learning_rates = [1e-6]
num_epochs_list = [300]

# Initialize an index counter
index = 1
accuracy_arr = []
# Iterate over parameter combinations

for lr in learning_rates:
    for num_epochs in num_epochs_list:
        print("new graph starts here")
        print("the lr is:", lr)
        print("the batch size is:", numb_batch)
        print("the number of epochs is:", num_epochs)
        # Train the model with current parameter settings
        lenet, accuracy = train(train_dl, val_dl, num_epochs, lr, device=device)
        accuracy_arr.append(accuracy)



# You can evaluate the model's performance on test data or print relevant metrics

# Create a filename with the index and parameter values
# filename = f"lenet_{index}_lr_{lr}_epochs_{num_epochs}.pt"
#
# # Save the trained model with the index in the filename
# torch.save(lenet.state_dict(), filename)
#
# # Increment the index counter
# index += 1
#
# print(f"Model saved as: {filename}")

#
# plt.plot(accuracy_arr[0], label='1e-3 40')
# plt.plot(accuracy_arr[1], label='1e-3 60')
# plt.plot(accuracy_arr[2], label='1e-3 80')
# plt.plot(accuracy_arr[3], label='1e-4 40')
# plt.plot(accuracy_arr[4], label='1e-4 60')
# plt.plot(accuracy_arr[5], label='1e-4 80')
# plt.plot(accuracy_arr[6], label='1e-5 40')
# plt.plot(accuracy_arr[7], label='1e-5 60')
# plt.plot(accuracy_arr[8], label='1e-5 80')
# plt.legend()
# plt.show()

# numb_epoch = 40
# lr = 1e-4
# print("the lr is:" ,lr)
# print("the batch size is:", numb_batch)
# print("the number of epochs is:", numb_epoch)
# # lenet,accuracy = train(train_dl, val_dl, numb_epoch=numb_epoch, lr=lr, device=device)
# loss,accuracy = train(train_dl, val_dl, numb_epoch=numb_epoch, lr=lr, device=device)
#
# plt.plot(accuracy, label='acc: 1e-4 40')
# plt.legend()
# plt.show()

# torch.save(lenet.state_dict(), "lenet2.pth")

# lenet = create_lenet().to(device)
# lenet.load_state_dict(torch.load("lenet2.pth"))
# lenet.eval()
#
# y_pred, y_true = predict_dl(lenet, val_dl, device=device)

# pd.DataFrame(confusion_matrix(y_true, y_pred, labels=np.arange(0, 10)))

# path = "https://previews.123rf.com/images/aroas/aroas1704/aroas170400068/79321959-handwritten-sketch-black-number-8-on-white-background.jpg"
# r = requests.get(path)
# with BytesIO(r.content) as f:
#     img = Image.open(f).convert(mode="L")
#     img = img.resize((28, 28))
# x = (255 - np.expand_dims(np.array(img), -1)) / 255.
#
# plt.imshow(x.squeeze(-1), cmap="gray")
# plt.show()
#
# pred = inference(path, lenet, device=device)
# pred_idx = np.argmax(pred)
# print(f"Predicted: {pred_idx}, Prob: {pred[0][pred_idx] * 100} %")

from utils import *
from data import cryo_np_Dataset
from torch.utils.data import DataLoader

numb_batch = 64
MNIST = True
MNIST = False
print("MNIST:",MNIST)
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



    # Create custom datasets for train and validation
    train_data = cryo_np_Dataset(outliers_file_path, particles_file_path, train=True, transform=custom_transforms)
    val_data = cryo_np_Dataset(outliers_file_path, particles_file_path, train=False, transform=custom_transforms)

    # Create dataloaders
    train_dl = DataLoader(train_data, batch_size=numb_batch, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=numb_batch, shuffle=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Cuda Available!!!!!!!!")
else:
    device = torch.device("cpu")
    print("No Cuda Available")
lenet = train(train_dl, val_dl, numb_epoch=40, device=device)

torch.save(lenet.state_dict(), "lenet2.pth")

lenet = create_lenet().to(device)
lenet.load_state_dict(torch.load("lenet2.pth"))
lenet.eval()

y_pred, y_true = predict_dl(lenet, val_dl, device=device)

# pd.DataFrame(confusion_matrix(y_true, y_pred, labels=np.arange(0, 10)))

path = "https://previews.123rf.com/images/aroas/aroas1704/aroas170400068/79321959-handwritten-sketch-black-number-8-on-white-background.jpg"
r = requests.get(path)
with BytesIO(r.content) as f:
    img = Image.open(f).convert(mode="L")
    img = img.resize((28, 28))
x = (255 - np.expand_dims(np.array(img), -1)) / 255.

plt.imshow(x.squeeze(-1), cmap="gray")
plt.show()

pred = inference(path, lenet, device=device)
pred_idx = np.argmax(pred)
print(f"Predicted: {pred_idx}, Prob: {pred[0][pred_idx] * 100} %")

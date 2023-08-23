from utils import *

numb_batch = 64

T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_data = torchvision.datasets.MNIST('mnist_data', train=True, download=True, transform=T)
val_data = torchvision.datasets.MNIST('mnist_data', train=False, download=True, transform=T)

train_dl = torch.utils.data.DataLoader(train_data, batch_size=numb_batch)
val_dl = torch.utils.data.DataLoader(val_data, batch_size=numb_batch)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Cuda Available!!!!!!!!")
else:
    device = torch.device("cpu")
    print("No Cuda Available")

lenet = train(40, device=device)

torch.save(lenet.state_dict(), "lenet.pth")
# lenet = create_lenet().to(device)
# lenet.load_state_dict(torch.load("lenet.pth"))
# lenet.eval()
#
# y_pred, y_true = predict_dl(lenet, val_dl, device=device)
#
# pd.DataFrame(confusion_matrix(y_true, y_pred, labels=np.arange(0, 10)))
#
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

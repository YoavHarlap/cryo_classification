from utils import *
#pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Cuda Available!!!!!!!!")
else:
    device = torch.device("cpu")
    print("No Cuda Available")

numb_batch = 64

T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# train_data = torchvision.datasets.MNIST('mnist_data', train=True, download=True, transform=T)
val_data = torchvision.datasets.MNIST('mnist_data', train=False, download=True, transform=T)

# train_dl = torch.utils.data.DataLoader(train_data, batch_size=numb_batch)
val_dl = torch.utils.data.DataLoader(val_data, batch_size=numb_batch)



lenet = create_lenet().to(device)
lenet.load_state_dict(torch.load("lenet.pth"))
lenet.eval()



y_pred, y_true = predict_dl(lenet, val_dl,device=device)
#
# pd.DataFrame(confusion_matrix(y_true, y_pred, labels=np.arange(0, 10)))
# # Calculate confusion matrix
# conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(0, 10))
#
# # Create a DataFrame for the confusion matrix
# conf_matrix_df = pd.DataFrame(conf_matrix)

import seaborn as sns


#  Create a heatmap using seaborn
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()


path = "https://previews.123rf.com/images/aroas/aroas1704/aroas170400068/79321959-handwritten-sketch-black-number-8-on-white-background.jpg"
path = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSbPo-8qulTB43XqKb8PmVYroIZet-2RqJneQ&usqp=CAU"
path = "https://saraai.com/images/blog/mnist1.png" #fail
path = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTQgEy8gd-Ydv5tnLL028Q1AAYLodrL_SMdj8GtlxYb3APJaQawP_103r_r1ufohknspy8&usqp=CAU"
path = "https://static.wixstatic.com/media/3b7e0c_aa83b027be574a7f91650f2ba307338e~mv2.png/v1/fill/w_548,h_545,al_c,lg_1,q_85,enc_auto/3b7e0c_aa83b027be574a7f91650f2ba307338e~mv2.png"
r = requests.get(path)
with BytesIO(r.content) as f:
    img = Image.open(f).convert(mode="L")
    img = img.resize((28, 28))
x = (255 - np.expand_dims(np.array(img), -1)) / 255.

# from scipy.ndimage import gaussian_filter
# x = gaussian_filter(x, sigma=3)
#

plt.imshow(x.squeeze(-1), cmap="gray")
plt.show()

pred = inference(path, lenet, device=device)
pred_idx = np.argmax(pred)
print(f"Predicted: {pred_idx}, Prob: {pred[0][pred_idx] * 100} %")

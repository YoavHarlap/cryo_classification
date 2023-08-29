# from utils import *
#
# numb_batch = 64
#
# T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# train_data = torchvision.datasets.MNIST('mnist_data', train=True, download=True, transform=T)
# val_data = torchvision.datasets.MNIST('mnist_data', train=False, download=True, transform=T)
#
# train_dl = torch.utils.data.DataLoader(train_data, batch_size=numb_batch)
# val_dl = torch.utils.data.DataLoader(val_data, batch_size=numb_batch)
#
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("Cuda Available!!!!!!!!")
# else:
#     device = torch.device("cpu")
#     print("No Cuda Available")
#
# lenet = train(40, device=device)
#
# torch.save(lenet.state_dict(), "lenet.pth")
# # lenet = create_lenet().to(device)
# # lenet.load_state_dict(torch.load("lenet.pth"))
# # lenet.eval()
# #
# # y_pred, y_true = predict_dl(lenet, val_dl, device=device)
# #
# # pd.DataFrame(confusion_matrix(y_true, y_pred, labels=np.arange(0, 10)))
# #
# # path = "https://previews.123rf.com/images/aroas/aroas1704/aroas170400068/79321959-handwritten-sketch-black-number-8-on-white-background.jpg"
# # r = requests.get(path)
# # with BytesIO(r.content) as f:
# #     img = Image.open(f).convert(mode="L")
# #     img = img.resize((28, 28))
# # x = (255 - np.expand_dims(np.array(img), -1)) / 255.
# #
# # plt.imshow(x.squeeze(-1), cmap="gray")
# # plt.show()
# #
# # pred = inference(path, lenet, device=device)
# # pred_idx = np.argmax(pred)
# # print(f"Predicted: {pred_idx}, Prob: {pred[0][pred_idx] * 100} %")


import re
import matplotlib.pyplot as plt

# Provided text
text = """
/home/yoavharlap/miniconda3/envs/cryo_classification/bin/python /home/yoavharlap/PycharmProjects/cryo_classification/train.py
MNIST: False
Cuda Available!!!!!!!!
Saving Best Model with Accuracy:  72.2188949584961
Epoch: 1 / 40 Accuracy : 72.2188949584961 % loss: 0.7083390355110168
Saving Best Model with Accuracy:  72.23524475097656
Epoch: 2 / 40 Accuracy : 72.23524475097656 % loss: 0.6573639512062073
Epoch: 3 / 40 Accuracy : 72.2188949584961 % loss: 0.5611185431480408
Saving Best Model with Accuracy:  74.3445816040039
Epoch: 4 / 40 Accuracy : 74.3445816040039 % loss: 0.3657324016094208
Saving Best Model with Accuracy:  78.1108627319336
Epoch: 5 / 40 Accuracy : 78.1108627319336 % loss: 0.328195184469223
Epoch: 6 / 40 Accuracy : 73.51610565185547 % loss: 0.5025725364685059
Epoch: 7 / 40 Accuracy : 77.90374755859375 % loss: 0.2861538529396057
Epoch: 8 / 40 Accuracy : 77.31509399414062 % loss: 0.2724774479866028
Epoch: 9 / 40 Accuracy : 72.91654968261719 % loss: 0.4007415473461151
Epoch: 10 / 40 Accuracy : 75.72900390625 % loss: 0.36671149730682373
Epoch: 11 / 40 Accuracy : 75.70175170898438 % loss: 0.1701437085866928
Epoch: 12 / 40 Accuracy : 73.69596862792969 % loss: 0.12891346216201782
Epoch: 13 / 40 Accuracy : 67.6513900756836 % loss: 0.1670914590358734
Epoch: 14 / 40 Accuracy : 67.30255889892578 % loss: 0.1635880023241043
Epoch: 15 / 40 Accuracy : 73.01466369628906 % loss: 0.21192960441112518
Epoch: 16 / 40 Accuracy : 73.62511444091797 % loss: 0.18277525901794434
Epoch: 17 / 40 Accuracy : 69.64081573486328 % loss: 0.3342669606208801
Epoch: 18 / 40 Accuracy : 71.3359146118164 % loss: 0.31428369879722595
Epoch: 19 / 40 Accuracy : 68.02201843261719 % loss: 0.31671613454818726
Epoch: 20 / 40 Accuracy : 71.65203857421875 % loss: 0.2562897801399231
Epoch: 21 / 40 Accuracy : 66.55583953857422 % loss: 0.25244662165641785
Epoch: 22 / 40 Accuracy : 72.70943450927734 % loss: 0.14157991111278534
Epoch: 23 / 40 Accuracy : 68.39810180664062 % loss: 0.2385876476764679
Epoch: 24 / 40 Accuracy : 69.17752075195312 % loss: 0.08378823846578598
Epoch: 25 / 40 Accuracy : 68.12557983398438 % loss: 0.2920045256614685
Epoch: 26 / 40 Accuracy : 66.80110931396484 % loss: 0.18317389488220215
Epoch: 27 / 40 Accuracy : 70.24581909179688 % loss: 0.09435553848743439
Epoch: 28 / 40 Accuracy : 68.7578353881836 % loss: 0.16959890723228455
Epoch: 29 / 40 Accuracy : 70.30577087402344 % loss: 0.16137424111366272
Epoch: 30 / 40 Accuracy : 70.8835220336914 % loss: 0.05735362321138382
Epoch: 31 / 40 Accuracy : 69.05216217041016 % loss: 0.21412669122219086
Epoch: 32 / 40 Accuracy : 67.01913452148438 % loss: 0.020664526149630547
Epoch: 33 / 40 Accuracy : 68.63247680664062 % loss: 0.13643552362918854
Epoch: 34 / 40 Accuracy : 68.56707000732422 % loss: 0.21591395139694214
Epoch: 35 / 40 Accuracy : 68.71968078613281 % loss: 0.05412813648581505
Epoch: 36 / 40 Accuracy : 72.59497833251953 % loss: 0.04142003878951073
Epoch: 37 / 40 Accuracy : 70.15860748291016 % loss: 0.12901999056339264
Epoch: 38 / 40 Accuracy : 69.16661834716797 % loss: 0.09540185332298279
Epoch: 39 / 40 Accuracy : 69.83702850341797 % loss: 0.007646183017641306
Epoch: 40 / 40 Accuracy : 70.25672149658203 % loss: 0.050944484770298004

Process finished with exit code 0

"""

# Extract epoch numbers, loss values, and accuracy values using regex
epochs = re.findall(r"Epoch: (\d+) / \d+ Accuracy : ([\d.]+) % loss: ([\d.]+)", text)

# Convert extracted data to lists
epoch_numbers, accuracies, loss_values = zip(*epochs)
epoch_numbers = list(map(int, epoch_numbers))
accuracies = list(map(float, accuracies))
loss_values = list(map(float, loss_values))

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Plot for Losses
axs[0].plot(epoch_numbers, loss_values, marker='o')
axs[0].set_title('Losses Over Epochs')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].grid(True)

# Plot for Accuracy
axs[1].plot(epoch_numbers, accuracies, marker='o', color='orange')
axs[1].set_title('Accuracy Over Epochs')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy (%)')
axs[1].grid(True)

plt.tight_layout()
plt.show()

#
# # Extract epoch numbers and loss values using regex
# epochs = re.findall(r"Epoch: (\d+) / \d+ Accuracy : [\d.]+ % loss: ([\d.]+)", text)
#
# # Convert extracted data to lists
# epoch_numbers, loss_values = zip(*epochs)
# epoch_numbers = list(map(int, epoch_numbers))
# loss_values = list(map(float, loss_values))
#
# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(epoch_numbers, loss_values, marker='o')
# plt.title('Losses Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.show()
#


text = """/home/yoavharlap/miniconda3/envs/cryo_classification/bin/python /home/yoavharlap/PycharmProjects/cryo_classification/train.py
MNIST: False
Cuda Available!!!!!!!!
new graph start here
the lr is: 0.001
the batch size is: 64
the number of epochs is: 2
Saving Best Model with Accuracy:  71.5321273803711
Epoch: 1 / 2 Accuracy : 71.5321273803711 % loss: 0.48366907238960266
Epoch: 2 / 2 Accuracy : 71.5321273803711 % loss: 0.6806074976921082
new graph start here
the lr is: 0.001
the batch size is: 64
the number of epochs is: 2
Saving Best Model with Accuracy:  71.5321273803711
Epoch: 1 / 2 Accuracy : 71.5321273803711 % loss: 0.5407228469848633
Epoch: 2 / 2 Accuracy : 71.5321273803711 % loss: 0.7192851901054382
new graph start here
the lr is: 0.001
the batch size is: 64
the number of epochs is: 2
Saving Best Model with Accuracy:  71.5321273803711
Epoch: 1 / 2 Accuracy : 71.5321273803711 % loss: 0.5929076075553894
Epoch: 2 / 2 Accuracy : 71.5321273803711 % loss: 0.4698238670825958
new graph start here
the lr is: 0.0001
the batch size is: 64
the number of epochs is: 2
Saving Best Model with Accuracy:  70.1041030883789
Epoch: 1 / 2 Accuracy : 70.1041030883789 % loss: 0.7379618287086487
Saving Best Model with Accuracy:  71.5321273803711
Epoch: 2 / 2 Accuracy : 71.5321273803711 % loss: 0.5938416719436646
new graph start here
the lr is: 0.0001
the batch size is: 64
the number of epochs is: 2
Saving Best Model with Accuracy:  71.1178970336914
Epoch: 1 / 2 Accuracy : 71.1178970336914 % loss: 0.6137155294418335
Epoch: 2 / 2 Accuracy : 55.74753189086914 % loss: 0.5513162016868591
new graph start here
the lr is: 0.0001
the batch size is: 64
the number of epochs is: 2
Saving Best Model with Accuracy:  71.51577758789062
Epoch: 1 / 2 Accuracy : 71.51577758789062 % loss: 0.9690212607383728
Saving Best Model with Accuracy:  77.6584701538086
Epoch: 2 / 2 Accuracy : 77.6584701538086 % loss: 0.5328044295310974
new graph start here
the lr is: 1e-05
the batch size is: 64
the number of epochs is: 2
Saving Best Model with Accuracy:  28.46786880493164
Epoch: 1 / 2 Accuracy : 28.46786880493164 % loss: 15.853089332580566
Saving Best Model with Accuracy:  71.5321273803711
Epoch: 2 / 2 Accuracy : 71.5321273803711 % loss: 3.564964771270752
"""


def show_misclassified_images(test_loader, model, device, max_display=5):
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

        if predict_y.shape[0] == 0:
            print("Predictions array is empty.")
        else:
            print("misclassified_indices.shape",misclassified_indices.shape)
            print("misclassified_indices.numel()",misclassified_indices.numel())
            if misclassified_indices.numel() == 0:
                print("Misclassified indices array is empty.")
            else:
                print("Misclassified indices array is not empty.")
                print("misclassified_indices", misclassified_indices)
                misclassified_images.extend(test_x[misclassified_indices])

                if predict_y[misclassified_indices].dim() == 0:  # Check if scalar
                    misclassified_labels.extend([predict_y[misclassified_indices].item()])
                    print("shape_scalar:", len([predict_y[misclassified_indices].item()]))
                else:
                    misclassified_labels.extend(predict_y[misclassified_indices].cpu().numpy())
                    print("shape:",predict_y[misclassified_indices].cpu().numpy().shape)


        correct_images.extend(test_x[correct_indices])
        correct_labels.extend(predict_y[correct_indices].cpu().numpy())

        # if len(misclassified_images) >= max_display and len(correct_images) >= max_display:
        #     break

    if len(misclassified_images) > 0:
        misclassified_images = torch.stack(misclassified_images)
        correct_images = torch.stack(correct_images)
        plt.figure(figsize=(15, 8))
        max_display = min(max_display,len(misclassified_images))
        print(misclassified_indices)
        print(misclassified_indices.shape)

        for i in range(max_display):
            plt.subplot(2, max_display, i + 1)
            plt.imshow(misclassified_images[i].cpu().numpy().squeeze(), cmap='gray')
            plt.title(
                f"Misclassified\nPred: {misclassified_labels[i]}, Correct: {test_label[misclassified_indices][i]}")

            plt.axis('off')

            plt.subplot(2, max_display, i + 1 + max_display)
            plt.imshow(correct_images[i].cpu().numpy().squeeze(), cmap='gray')
            plt.title(f"Correctly Classified\nPred: {correct_labels[i]}, Correct: {test_label[correct_indices][i].item()}")
            plt.axis('off')
    else:
        print("No misclassified images to display.")

    plt.tight_layout()
    plt.savefig('misclassified_images.png')
    plt.show()

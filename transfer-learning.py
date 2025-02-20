import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from clearml import Task, Logger

def main():

    model_snapshots_path = './clearml'
    if not os.path.exists(model_snapshots_path):
        os.makedirs(model_snapshots_path)

    task = Task.init(project_name='custom',
                     task_name='PyTorch Custom train',
                     output_uri=model_snapshots_path)
    logger = task.get_logger

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5,))                                
    ])


    train_dataset = datasets.ImageFolder("./hymenoptera_data/train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = datasets.ImageFolder("./hymenoptera_data/val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # data_iter = iter(train_loader)
    # imgs, labels = next(data_iter)
    # imgs.size()
    # img = imgs[0]
    # img_permute = img.permute(1, 2, 0)
    # img_permute = 0.5 * img_permute + 0.5
    # img_permute = np.clip(img_permute, 0, 1)
    # plt.imshow(img_permute)


    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False


    model.fc = nn.Linear(512, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    num_epochs = 16
    losses = []
    accs = []
    val_losses = []
    val_accs = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            running_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            running_acc += torch.mean(pred.eq(labels).float())
            loss.backward()
            optimizer.step()
        running_loss /= len(train_loader)
        running_acc /= len(train_loader)
        losses.append(running_loss)
        # accs.append(running_acc)
        accs.append(running_acc.cpu())

        #
        # validation loop
        #
        val_running_loss = 0.0
        val_running_acc = 0.0
        for val_imgs, val_labels in val_loader:
            val_imgs = val_imgs.to(device)
            val_labels = val_labels.to(device)
            val_output = model(val_imgs)
            val_loss = criterion(val_output, val_labels)
            val_running_loss += val_loss.item()
            pred = torch.argmax(val_output, dim=1)
            val_running_acc += torch.mean(pred.eq(val_labels).float())
            # loss.backward()
            # optimizer.step()
        val_running_loss /= len(val_loader)
        val_running_acc /= len(val_loader)
        val_losses.append(val_running_loss)
        # accs.append(val_running_acc)
        val_accs.append(val_running_acc.cpu())

        # print("epoch: {}, loss: {}, acc: {}".format(epoch, running_loss, running_acc))
        print("epoch: {}, loss: {}, acc: {}, \
        val loss: {}, val acc: {}".format(epoch, running_loss, running_acc, val_running_loss, val_running_acc))


    # plt.plot(losses)
    # plt.legend()
    # plt.xlabel("Epochs")
    # plt.ylabel("Error")
    # plt.show()

    # plt.plot(accs)
    # plt.legend()
    # plt.xlabel("Epochs")
    # plt.ylabel("Error")
    # plt.show()

    # Save model
    model_scripted = torch.jit.script(model)
    model_scripted.save('model_scripted.pth')

if __name__ == '__main__':
    main()





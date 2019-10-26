import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import json
import sys

data_transform = transforms.Compose(
        [transforms.RandomApply([
        transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.01),
        transforms.RandomRotation(50),
        transforms.RandomResizedCrop(224, scale = (0.4, 1.0)),
        transforms.RandomHorizontalFlip()], p = 0.4),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

fp = sys.argv[1] if len(sys.argv) > 1 else 'data-example/'
data_dir = fp + 'train'
image_dataset = datasets.ImageFolder(data_dir, data_transform)
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=4)


dataset_size = len(image_dataset)
class_names = image_dataset.classes

json.dump(class_names, open("class_names.json", "w"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    torch.save(best_model_wts, "models/best_so_far.pt")
    best_loss = None

    losses = []

    #test_inference(model)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        displayed = False

        # Iterate over data.
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if (not displayed):
                imgs = torchvision.utils.make_grid(inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if (not displayed):
                    imshow(imgs, title=[class_names[x] for x in preds])
                    displayed = True

                # backward + optimize only if in training phase
                
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        scheduler.step()

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        losses.append(epoch_loss)

        # deep copy the model
        if best_loss is None or epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "models/best_so_far.pt")

        if (len(losses) >= 4 and min(losses[-3:-1]) > losses[-4] * 0.95):
            print("Early Stopping")
            break

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


custom_num_classes = len(class_names)

model_conv = torchvision.models.squeezenet1_1(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

model_conv.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Conv2d(512, custom_num_classes, kernel_size=1),
    nn.ReLU(inplace=True),
    nn.AvgPool2d(13)
)
model_conv.forward = lambda x: model_conv.classifier(model_conv.features(x)).view(x.size(0), custom_num_classes)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.classifier.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.5 every 3 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=3, gamma=0.5)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)





import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import time
import os
import copy
import json
import sys

if (sys.argv[1] == 'train'):

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

    fp = sys.argv[2]
    data_dir = fp
    image_dataset = datasets.ImageFolder(data_dir, data_transform)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=4)


    dataset_size = len(image_dataset)
    class_names = image_dataset.classes

    class_names_path = f"class_names/{sys.argv[3]}_classes.json"
    json.dump(class_names, open(class_names_path, "w"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        save_path = f"models/{sys.argv[3]}_params.pt"

        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, save_path)
        best_loss = None

        losses = []

        #test_inference(model)

        for epoch in range(num_epochs):
            print('epoch {} {}'.format(epoch+1, num_epochs))

            model.train()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase

                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            scheduler.step()

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            # print('loss-acc {:.4f} {:.4f}'.format(epoch_loss, epoch_acc))

            losses.append(epoch_loss)

            # deep copy the model
            if best_loss is None or epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, save_path)

            if (len(losses) >= 4 and min(losses[-3:-1]) > losses[-4] * 0.95):
                break

        time_elapsed = time.time() - since
        print('done');
        # print('done {:f} {:4f}'.format(time_elapsed, best_loss))

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
elif (sys.argv[1] == 'infer'):
    device = torch.device('cpu')

    load_path = f"models/{sys.argv[2]}_params.pt"
    class_names_path = f"class_names/{sys.argv[2]}_classes.json"

    def load_model(num_classes):
        custom_num_classes = num_classes

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

        model_conv.load_state_dict(torch.load(load_path))
        return model_conv

    def load_data(path):
        data_transforms = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        image = Image.open(path).convert('RGB')
        image = data_transforms(image)
        image = image.unsqueeze(0)

        return image


    def inference(model, image):
        model.eval()
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        exp_outputs = np.exp(outputs.detach().numpy())
        normed_outputs = exp_outputs / np.sum(exp_outputs, axis = 1, keepdims = True)
        return predicted.numpy()[0], normed_outputs[0][predicted[0]]

    class_names = json.load(open(class_names_path, "r"))
    model = load_model(len(class_names))

    for line in sys.stdin:
        path = line.rstrip()

        image = load_data(path)
        prediction, confidence = inference(model, image)
        print(f'res {path} {class_names[prediction]} {confidence}', flush=True)

        sys.stdout.flush()
else:
    print('Unknown command', flush=True)

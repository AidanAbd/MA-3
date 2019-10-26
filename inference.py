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
device = torch.device('cpu')

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

	model_conv.load_state_dict(torch.load('models/best_so_far.pt'))
	return model_conv

def load_data(path):
	data_transforms = transforms.Compose([
				        transforms.Resize(256),
				        transforms.CenterCrop(224),
				        transforms.ToTensor(),
				        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	test_images = datasets.ImageFolder(path, data_transforms)
	test_loader = torch.utils.data.DataLoader(test_images, batch_size=len(test_images),
                                             shuffle=False, num_workers=4)

	dataiter = iter(test_loader)
	images, labels = dataiter.next()
	#imshow(torchvision.utils.make_grid(images))

	return images


def inference(model, images):
	model.eval()
	outputs = model(images)
	_, predicted = torch.max(outputs, 1)
	return predicted.numpy()

def post_process(predictions, path):
	class_names = json.load(open("class_names.json", "r"))
	image_names = sorted([p for p in os.listdir(path) if '.jpg' in p])

	processed = dict()
	for img_name, p in zip(image_names, predictions):
		processed[img_name] = class_names[p]
	return processed


model = load_model(2)
images = load_data('data/hymenoptera_data/test')
predictions = inference(model, images)
print(post_process(predictions, 'data/hymenoptera_data/test/unlabeled'))


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

test_images = image_datasets['test']
test_loader = torch.utils.data.DataLoader(test_images, batch_size=len(test_images),
                                             shuffle=False, num_workers=4)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

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
    best_acc = 0.0

    #test_inference(model)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)



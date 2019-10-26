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

data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir = 'data-example/train'
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


# Get a batch of training data
inputs, classes = next(iter(dataloader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    if (len(sys.argv) == 1 or sys.argv[1] == "new"):
        torch.save(best_model_wts, "models/best_so_far.pt")
    elif (sys.argv[1] == "prev"):
        model.load_state_dict(torch.load('models/best_so_far.pt'))
    best_acc = 0.0

    #test_inference(model)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        
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

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "models/best_so_far.pt")

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
'''
def test_inference(model):
    
    #for inputs, labels in test_loader:

    (inputs, labels) = iter(test_loader).next()

    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    print(os.listdir("data/hymenoptera_data/test/unlabeled"))
    #print(os.listdir("data/hymenoptera_data/test/ants"))
    #print(os.listdir("data/hymenoptera_data/test/bees"))
    imshow(inputs[0])
    time.sleep(1)
    imshow(inputs[1])
    time.sleep(1)
    imshow(inputs[2])
    time.sleep(1)
    print(torch.mean(inputs))
    print(preds)
'''
'''
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
'''

#model_conv = torchvision.models.squeezenet1_0(pretrained=True)
#for param in model_conv.parameters():
#    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
#num_ftrs = model_conv.classifier.in_features
#num_ftrs = 512
#model_conv.classifier[1] = nn.Conv2d(512, 2, kernel_size = (1, 1), stride = (1, 1))

'''
class transferred_squeeze(nn.Module):

	def __init__(self):
		super().__init__()

		self.squeeze_net = torchvision.models.squeezenet1_0(pretrained=True)
		for param in self.squeeze_net.parameters():
			param.requires_grad = False
		self.fc = nn.Linear(1000, 2)

	def forward(self, x):

		squeeze_net_out = self.squeeze_net(x)
		transfer_out = self.fc(squeeze_net_out)
		return transfer_out
'''

#model_conv.classifier.add_module("transpose", transposer())
#model_conv.classifier.add_module("fc", nn.Linear(4000, 2))

#model_conv = transferred_squeeze()

custom_num_classes = 2

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

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)





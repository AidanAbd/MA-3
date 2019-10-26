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

	model_conv.load_state_dict(torch.load('models/best_so_far'))
	return model_conv

def load_data(testloader):
	dataiter = iter(testloader)
	images, labels = dataiter.next()
	imshow(torchvision.utils.make_grid(images))


def inference(model, images):
	model.eval()
	outputs = model(images)
	_, predicted = torch.max(outputs, 1)
	return predicted


model = load_model(2)
load_data('''dataloader goes here''')
predictions = inference(model, images)
print(predictions)

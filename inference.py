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
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
device = torch.device('cpu')

fp = sys.argv[1] if len(sys.argv) == 2 else 'data-example/'

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
	exp_outputs = np.exp(outputs.detach().numpy())
	normed_outputs = exp_outputs / np.sum(exp_outputs, axis = 1, keepdims = True)
	confidences = [normed_outputs[i][predicted[i]] for i in range(len(images))]
	return {"classes": predicted.numpy(), "confidences": confidences}

def post_process(predictions, path):
	is_letter = lambda c: ord(c.lower()) >= ord('a') and ord(c.lower()) <= ord('z')
	image_names = sorted([p for p in os.listdir(path) if is_letter(p[0])])

	processed = dict()
	correct = 0
	correct_conf = 0
	for img_name, p, c in zip(image_names, predictions["classes"], predictions["confidences"]):
		processed[img_name] = class_names[p]
		if (class_names[p].lower() in img_name.lower()):
			correct += 1
			correct_conf += c
	print(f"ACCURACY: {correct * 100 / len(image_names):.2f}%")
	print(f"AVG CORRECT CONFIDENCE: {correct_conf * 100 / correct:.2f}%")
	return processed

class_names = json.load(open("class_names.json", "r"))
model = load_model(len(class_names))
images = load_data(fp + 'test')
predictions= inference(model, images)
print(post_process(predictions, fp + 'test/unlabeled'))



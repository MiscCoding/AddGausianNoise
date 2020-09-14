# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint
from skimage.util import random_noise
from scipy import signal
import torch.distributions as tdist
import math
from PIL import Image

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from torch.nn.utils import vector_to_parameters, parameters_to_vector

print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot



parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default="5000",
                    help='the index for leaking images on CIFAR.')
parser.add_argument('--image', type=str, default="",
                    help='the path to customized image.')
args = parser.parse_args()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

dst = datasets.CIFAR100("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

img_index = args.index
gt_data = tp(dst[img_index][0]).to(device)

if len(args.image) > 1:
    gt_data = Image.open(args.image)
    gt_data = tp(gt_data).to(device)

gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
gt_label = gt_label.view(1, )
gt_onehot_label = label_to_onehot(gt_label)

plt.imshow(tt(gt_data[0].cpu()))

from models.vision import LeNet, weights_init

net = LeNet().to(device)

torch.manual_seed(1234)

net.apply(weights_init)
criterion = cross_entropy_for_onehot

# compute original gradient
pred = net(gt_data)
y = criterion(pred, gt_onehot_label)



# Gausian Noise adder
def addGausianNoise(standardDeviation=[1.0]):
    # Gausian Laplace Noise
    param_vector = parameters_to_vector(net.parameters())
    n_params = len(param_vector)
    noise = tdist.Normal(torch.tensor([0.0]), torch.tensor(standardDeviation)).sample(sample_shape=torch.Size([n_params, ]))
    print(noise)
    noiseExtracted = np.array([])
    for counter, value in enumerate(noise):
        noiseExtracted = np.append(noiseExtracted, value)

    noiseExtracted = torch.from_numpy(noiseExtracted)

    param_vector.add_(noiseExtracted)
    vector_to_parameters(param_vector, net.parameters())


def accuracy(model, testloader, device="cuda", max_total=-1):
    correct = 0.0
    total = 0.0

    if max_total == -1:
        max_total = 999999

    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            x_ts, y_ts = data
            # print(device)
            x_ts = x_ts.to(device)
            y_ts = y_ts.to(device)
            output = model(x_ts)
            _, predicted = torch.max(output.data, 1)

            # _, predicted = torch.argmax(output.data, 1)
            total += y_ts.size(0)
            correct += (predicted == y_ts).sum().item()
            if total > max_total:
                break
        return correct / total


## Accuray computation!!
def AreDummyAndOriginalLabelNoMatch(gt_oneshot_label, dummy_onehot_label_copy, decimalRound=3):
    originalLabelNo = int(torch.argmax(gt_oneshot_label[0]))
    dummyLabelCopy = int(torch.argmax(dummy_onehot_label_copy[0]))

    if(originalLabelNo == dummyLabelCopy):
        print("Correct!  The label no. is " + str(originalLabelNo))



    # print("Original Image is No." + str(pred) + ". The class : " + str(label))
    # print("Original length is " + str(dy_dx_roundedLength) + ". dummy length is " + str(dummy_dy_dx_roundedLength))
    #

    # dy_dx_rounded = (dy_dx * 10 ** decimalRound).round() / (10**decimalRound)
    # dummy_dy_dx_rounded = (dummy_dy_dx_copy * 10 ** decimalRound).round() / (10**decimalRound)
    # dy_dx_roundedLength = len(dy_dx_rounded[0])
    # dummy_dy_dx_roundedLength = len(dummy_dy_dx_rounded[0])
    # #
    # score = 0
    # print("Original Image is No." + str(dy_dx_rounded[0]) + ". The class : " + str(dummy_dy_dx_rounded[0]))
    # print("Original length is " + str(dy_dx_roundedLength) + ". dummy length is " + str(dummy_dy_dx_roundedLength))
    #
    # for original_Value, dummy_Value in zip(dy_dx_rounded, dummy_dy_dx_rounded):
    #     if (original_Value == dummy_Value):
    #         print("Original and dummy value " + original_Value + ". Dummy value is " + dummy_Value)
    #         score += 1
    #
    # accuray
    # print("Accuracy : " + score)


    ##print("Original Image is No." + str(dy_dx[0]) + ". The class : " + str(dummy_dy_dx_copy[0]))

    # print("Original list length = " + str(len(original)) + ". Dummy list length = " +str(len(dummy)))
    # accuracyScore = 0
    # originalList = original[0]
    # dummyList = dummy[0]
    #
    # totalSizeOriginalList = len(originalList) * len(originalList[0])
    # print("The orignal list size is " + str(totalSizeOriginalList))
    # outerListIndex = 0
    # print(originalList)
    # print(dummyList)
    # for outerIndex, outerValue in enumerate(originalList):
    #     outerListIndex = outerIndex
    #
    #     for innerIndex, innerValue in enumerate(outerValue):
    #         maxValueOfOriginal = max(innerValue)
    #         maxValueOfDummy = max(dummyList[outerIndex][innerIndex])
    #         if (maxValueOfDummy == maxValueOfOriginal):
    #             accuracyScore += 1
    #             print("Value match! score is " + str(accuracyScore))
    #
    #
    #     # maxValueOfOriginal = max(originalList[index])
    #     # maxValueOfDummy = max(dummyList[index])
    #     # if(maxValueOfDummy == maxValueOfOriginal):
    #     #     accuracyScore += 1
    #     #     print("Value match! score is " + str(accuracyScore))
    #
    # print("Accuracy score Dummy/Original is " + str(accuracyScore/totalSizeOriginalList))


##Image Label checker
def ImageLabelPrintoftheOriginalAndDummyOutput(gt_label, dummy_label):
    print("Original Image is No." + str(gt_label[0]) + ". Dummy image values : " + dst.classes[gt_label[0]])

def showImageLabel(dummy_pred_copy):
    print(dummy_pred_copy)



addNoiseOrNot = input("Please enter 'Y' to add Gaussian noise to the weight")

if addNoiseOrNot.lower() == 'Y'.lower():
    setVariation = input("Please enter a float number to add Variabtion to the weight")
    setVariation = float(setVariation)
    addGausianNoise([setVariation])
    print("Gaussian noise has been added")
else:
    print("No Gaussian noise has been added")


dy_dx = torch.autograd.grad(y, net.parameters())

original_dy_dx = list((_.detach().clone() for _ in dy_dx))

# generate dummy data and label
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

plt.imshow(tt(dummy_data[0].cpu()))

optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

history = []
dummy_label_copy = []
dummy_pred_copy = []
dummy_onehot_label_copy = []
dummy_loss_copy = []
dummy_dy_dx_copy = []

for iters in range(300):
    def closure():
        global dummy_label_copy
        global dummy_pred_copy
        global dummy_onehot_label_copy
        global dummy_loss_copy
        global dummy_dy_dx_copy

        optimizer.zero_grad()

        dummy_pred = net(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        #dummy_loss_copy  = dummy_loss

        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()

        dummy_label_copy = dummy_label
        dummy_pred_copy = dummy_pred
        dummy_onehot_label_copy = dummy_onehot_label
        dummy_loss_copy = dummy_loss
        dummy_dy_dx_copy = dummy_dy_dx


        return grad_diff


    optimizer.step(closure)
    if iters % 10 == 0:
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
        history.append(tt(dummy_data[0].cpu()))



plt.figure(figsize=(12, 8))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')


# ImageLabelPrintoftheOriginalAndDummyOutput(pred, dummy_pred_copy)

AreDummyAndOriginalLabelNoMatch(gt_onehot_label, dummy_onehot_label_copy)
plt.show()

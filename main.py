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
from models.vision import LeNet, weights_init



print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot

batch = 64

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default="20",
                    help='the index for leaking images on CIFAR.')
parser.add_argument('--image', type=str, default="",
                    help='the path to customized image.')
args = parser.parse_args()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)


totalAccuracy = 0

# dst = datasets.CIFAR100("~/.torch", download=True)
dst = datasets.CIFAR10("~/.torch", download=True)
imageDataBatchLoad = torch.utils.data.DataLoader(dst, batch_size=batch, shuffle=True, num_workers=2)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

# img_index = args.index
# gt_data = tp(dst[img_index][0]).to(device)

if len(args.image) > 1:
    gt_data1 = Image.open(args.image)
    gt_data1 = tp(gt_data1).to(device)

# from models.vision import LeNet, weights_init

net = LeNet().to(device)

## Accuray computation!!
def AreDummyAndOriginalLabelNoMatch(gt_oneshot_label, dummy_onehot_label_copy, decimalRound=3):
    originalLabelNo = int(torch.argmax(gt_oneshot_label[0]))
    dummyLabelCopy = int(torch.argmax(dummy_onehot_label_copy[0]))

    if(originalLabelNo == dummyLabelCopy):
        print("Dummy has successfully been restored!  The label no. is " + str(originalLabelNo))
        return True
    else:
        print("Restored from Image from Dummy is incorrect!  The original label no. is " + str(originalLabelNo) +". The dummy label no. is " + str(dummyLabelCopy))
        return False

# Gausian Noise adder
def addGausianNoise(net, standardDeviation=[1.0]):
    # Gausian Laplace Noise
    param_vector = parameters_to_vector(net.parameters())
    n_params = len(param_vector)
    noise = tdist.Normal(torch.tensor([0.0]), torch.tensor(standardDeviation)).sample(sample_shape=torch.Size([n_params, ]))
    #print(noise)
    noiseExtracted = np.array([])
    for counter, value in enumerate(noise):
        noiseExtracted = np.append(noiseExtracted, value)

    noiseExtracted = torch.from_numpy(noiseExtracted)

    param_vector.add_(noiseExtracted)
    vector_to_parameters(param_vector, net.parameters())

def DoesRestoreDummyImageFromGivenOriginalGradientSuccessful(net, weights_init, dst, indexNo,addNoiseOrNot="N", setVariation=0.1, showImageOnScreen="N", lossThresholdValueToJudgeInCorrect=1.00, judge_by_category_classification="Y"):
    gt_data = tp(dst[indexNo][0]).to(device)
    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = torch.Tensor([dst[indexNo][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label)
    plt.imshow(tt(gt_data[0].cpu()))

    # from models.vision import LeNet, weights_init
    #
    # net = LeNet().to(device)

    torch.manual_seed(1234)

    net.apply(weights_init)
    criterion = cross_entropy_for_onehot

    # compute original gradient
    pred = net(gt_data)
    y = criterion(pred, gt_onehot_label)
    if(addNoiseOrNot.lower() == "Y".lower()):
        setVariation = float(setVariation)
        addGausianNoise(net, [setVariation])

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
    lastDummyLossData = 0

    for iters in range(300):
        def closure():
            nonlocal dummy_label_copy
            nonlocal dummy_pred_copy
            nonlocal dummy_onehot_label_copy
            nonlocal dummy_loss_copy
            nonlocal dummy_dy_dx_copy
            nonlocal lastDummyLossData
            optimizer.zero_grad()

            dummy_pred = net(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
            # dummy_loss_copy  = dummy_loss

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
            lastDummyLossData = current_loss.item()
            history.append(tt(dummy_data[0].cpu()))

    plt.figure(figsize=(12, 8))
    for i in range(30):
        plt.subplot(3, 10, i + 1)
        plt.imshow(history[i])
        plt.title("iter=%d" % (i * 10))
        plt.axis('off')

    result = AreDummyAndOriginalLabelNoMatch(gt_onehot_label, dummy_onehot_label_copy)
    if(showImageOnScreen.lower() == "Y".lower()):
        plt.show()
        plt.close('all')
    if(lastDummyLossData > lossThresholdValueToJudgeInCorrect and (judge_by_category_classification.lower() == "N".lower())):
        print("Accuracy is being built by Dummy loss too")
        print("Dummy loss is still greater than the threshold."+ str(lossThresholdValueToJudgeInCorrect) +" Image restoration failed by the value " + str(lastDummyLossData))
        result = 0

    return int(result)


def AccuracyCalculationBasedOnNumberOfIteration(net_param, weights_init_param, sampleIndexUpto=20, addNoiseOrNot="N", setVariation=0.1, showImageOnScreen="N"):
    print("Iteration begins. It will take place "+ str(sampleIndexUpto) + "times.")
    totalHit = 0
    entirePopulationCount = sampleIndexUpto
    for img_indexInner in range(sampleIndexUpto):
        # gt_data_inner = tp(dst[img_indexInner][0]).to(device)
        totalHit += DoesRestoreDummyImageFromGivenOriginalGradientSuccessful(net_param, weights_init_param, dst, img_indexInner, addNoiseOrNot, setVariation, showImageOnScreen)

    print("DFG hacking made " + str(totalHit) + " out of " + str(entirePopulationCount) +" .")
    print("It scored " + str(float(100*(totalHit/entirePopulationCount))) + " percent.")


AccuracyCalculationBasedOnNumberOfIteration(net, weights_init, 10, "N", 0.1, "Y")

# gt_data = tp(dst[img_index][0]).to(device)
# hit = DoesRestoreDummyImageFromGivenOriginalGradientSuccessful(net, weights_init, gt_data)
# print("Hit restored : " + str(hit))
# img_index = args.index
# gt_data = tp(dst[img_index][0]).to(device)
#
# if len(args.image) > 1:
#     gt_data = Image.open(args.image)
#     gt_data = tp(gt_data).to(device)
#
# gt_data = gt_data.view(1, *gt_data.size())
# gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
# gt_label = gt_label.view(1, )
# gt_onehot_label = label_to_onehot(gt_label)
#
# plt.imshow(tt(gt_data[0].cpu()))
#
# from models.vision import LeNet, weights_init
#
# net = LeNet().to(device)
#
# torch.manual_seed(1234)
#
# net.apply(weights_init)
# criterion = cross_entropy_for_onehot
#
# # compute original gradient
# pred = net(gt_data)
# y = criterion(pred, gt_onehot_label)
#
#
# def accuracy(model, testloader, device="cuda", max_total=-1):
#     correct = 0.0
#     total = 0.0
#
#     if max_total == -1:
#         max_total = 999999
#
#     with torch.no_grad():
#         for i, data in enumerate(testloader, 0):
#             x_ts, y_ts = data
#             # print(device)
#             x_ts = x_ts.to(device)
#             y_ts = y_ts.to(device)
#             output = model(x_ts)
#             _, predicted = torch.max(output.data, 1)
#
#             # _, predicted = torch.argmax(output.data, 1)
#             total += y_ts.size(0)
#             correct += (predicted == y_ts).sum().item()
#             if total > max_total:
#                 break
#         return correct / total
#
#
#
#
#
# ##Image Label checker
# def ImageLabelPrintoftheOriginalAndDummyOutput(gt_label, dummy_label):
#     print("Original Image is No." + str(gt_label[0]) + ". Dummy image values : " + dst.classes[gt_label[0]])
#
# def showImageLabel(dummy_pred_copy):
#     print(dummy_pred_copy)
#
#
#
# addNoiseOrNot = input("Please enter 'Y' to add Gaussian noise to the weight")
#
# if addNoiseOrNot.lower() == 'Y'.lower():
#     setVariation = input("Please enter a float number to add Variabtion to the weight")
#     setVariation = float(setVariation)
#     addGausianNoise([setVariation])
#     print("Gaussian noise has been added")
# else:
#     print("No Gaussian noise has been added")
#
#
# dy_dx = torch.autograd.grad(y, net.parameters())
#
# original_dy_dx = list((_.detach().clone() for _ in dy_dx))
#
# # generate dummy data and label
# dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
# dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
#
# plt.imshow(tt(dummy_data[0].cpu()))
#
# optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
#
# history = []
# dummy_label_copy = []
# dummy_pred_copy = []
# dummy_onehot_label_copy = []
# dummy_loss_copy = []
# dummy_dy_dx_copy = []
#
# for iters in range(300):
#     def closure():
#         global dummy_label_copy
#         global dummy_pred_copy
#         global dummy_onehot_label_copy
#         global dummy_loss_copy
#         global dummy_dy_dx_copy
#
#         optimizer.zero_grad()
#
#         dummy_pred = net(dummy_data)
#         dummy_onehot_label = F.softmax(dummy_label, dim=-1)
#         dummy_loss = criterion(dummy_pred, dummy_onehot_label)
#         dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
#         #dummy_loss_copy  = dummy_loss
#
#         grad_diff = 0
#         for gx, gy in zip(dummy_dy_dx, original_dy_dx):
#             grad_diff += ((gx - gy) ** 2).sum()
#         grad_diff.backward()
#
#         dummy_label_copy = dummy_label
#         dummy_pred_copy = dummy_pred
#         dummy_onehot_label_copy = dummy_onehot_label
#         dummy_loss_copy = dummy_loss
#         dummy_dy_dx_copy = dummy_dy_dx
#
#
#         return grad_diff
#
#
#     optimizer.step(closure)
#     if iters % 10 == 0:
#         current_loss = closure()
#         print(iters, "%.4f" % current_loss.item())
#         history.append(tt(dummy_data[0].cpu()))
#
#
#
# plt.figure(figsize=(12, 8))
# for i in range(30):
#     plt.subplot(3, 10, i + 1)
#     plt.imshow(history[i])
#     plt.title("iter=%d" % (i * 10))
#     plt.axis('off')
#
#
# # ImageLabelPrintoftheOriginalAndDummyOutput(pred, dummy_pred_copy)
#
# AreDummyAndOriginalLabelNoMatch(gt_onehot_label, dummy_onehot_label_copy)
# plt.show()

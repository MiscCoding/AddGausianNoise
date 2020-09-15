import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from torch.nn.utils import vector_to_parameters, parameters_to_vector

### RUN CELL
# weight average model

t0 = time.time()

#### Parameters

batch = 64
filter_size = 64
config = [2, 2, 2, 2]

eps = 1e-10
num_models = 5

## Data setting

trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch * 10, shuffle=True, num_workers=2)
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

trainset_plain = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True,
                                              transform=test_transform)
trainloader_plain = torch.utils.data.DataLoader(trainset_plain, batch_size=batch, shuffle=True, num_workers=2)

## Model setting

criterion = torch.nn.CrossEntropyLoss()

datasize = int(len(trainset) / num_models)

sets = []
for i in range(num_models):
    sets.append(datasize)

subsets = torch.utils.data.random_split(trainset, sets, generator=torch.Generator().manual_seed(0))
subsets_plain = torch.utils.data.random_split(trainset_plain, sets, generator=torch.Generator().manual_seed(0))

models = []
opts = []
datasets = []
datasets_plain = []

for i in range(num_models):
    model = ResNet(kernel_size=filter_size, config=config, batch_size=batch, device=device)
    model.to(device)
    # model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    dataloader = torch.utils.data.DataLoader(subsets[i], batch_size=batch, shuffle=True, num_workers=2)

    models.append(model)
    opts.append(optimizer)
    datasets.append(dataloader)

central_model = ResNet(kernel_size=filter_size, config=config, batch_size=batch, device=device)
central_model.to(device)
central_opt = torch.optim.Adam(central_model.parameters(), lr=1e-4)

shadow_models_set_A = []
shadow_opts_set_A = []

for i in range(num_models):
    model = ResNet(kernel_size=filter_size, config=config, batch_size=batch, device=device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    shadow_models_set_A.append(model)
    shadow_opts_set_A.append(optimizer)

shadow_models_set_B = []
shadow_opts_set_B = []

for i in range(num_models):
    model = ResNet(kernel_size=filter_size, config=config, batch_size=batch, device=device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    shadow_models_set_B.append(model)
    shadow_opts_set_B.append(optimizer)

shadow_models_set_C = []
shadow_opts_set_C = []

for i in range(num_models):
    model = ResNet(kernel_size=filter_size, config=config, batch_size=batch, device=device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    shadow_models_set_C.append(model)
    shadow_opts_set_C.append(optimizer)

for i, model in enumerate(models):
    models[i].load_state_dict(central_model.state_dict())
    shadow_models_set_A[i].load_state_dict(central_model.state_dict())
    shadow_models_set_B[i].load_state_dict(central_model.state_dict())
    shadow_models_set_C[i].load_state_dict(central_model.state_dict())

### Test data setting to get std/ent


client_0_sample_data_iter = iter(datasets[0])
client_0_img_sample, client_0_label_sample = client_0_sample_data_iter.next()
client_0_img_sample = client_0_img_sample.to(device)
client_0_label_sample = client_0_label_sample.to(device)

client_1_sample_data_iter = iter(datasets[1])
client_1_img_sample, client_1_label_sample = client_1_sample_data_iter.next()
client_1_img_sample = client_1_img_sample.to(device)
client_1_label_sample = client_1_label_sample.to(device)

client_2_sample_data_iter = iter(datasets[2])
client_2_img_sample, client_2_label_sample = client_2_sample_data_iter.next()
client_2_img_sample = client_2_img_sample.to(device)
client_2_label_sample = client_2_label_sample.to(device)

model_0_data_0_layer = get_block_out(models[0], client_0_img_sample, device)
model_0_data_1_layer = get_block_out(models[0], client_1_img_sample, device)
model_0_data_0_out = models[0](client_0_img_sample)
model_0_data_1_out = models[0](client_1_img_sample)

# shadow_model_0_data_0_layer = get_block_out(shadow_model, client_0_img_sample, device)
# shadow_model_0_data_1_layer = get_block_out(shadow_model, client_1_img_sample, device)
# shadow_model_0_data_0_out = shadow_model(client_0_img_sample)
# shadow_model_0_data_1_out = shadow_model(client_1_img_sample)


###TEST


# output_shadow = get_block_out(shadow_model, client_1_img_sample, device)
# print("shape of output_shadow", np.shape(output_shadow))
# [layer, batch, n_kernel, w, h]


#### Acc init

total0 = 0.0
correct0 = 0.0
total1 = 0.0
correct1 = 0.0

_, predicted = torch.max(model_0_data_0_out.data, 1)
total0 += client_0_label_sample.size(0)
correct0 += (predicted == client_0_label_sample).sum().item()
acc0 = correct0 / total0

_, predicted = torch.max(model_0_data_1_out.data, 1)
total1 += client_1_label_sample.size(0)
correct1 += (predicted == client_1_label_sample).sum().item()
acc1 = correct1 / total1

ACC0 = []
ACC1 = []
ACC0.append(acc0)
ACC1.append(acc1)

# test_imgs = []
# for i in range(num_models):
#     data_iter = iter(datasets_plain[i])
#     img, _ = data_iter.next()
#     img = img.to(device)
#     print(img.size())
#     print(torch.max(img).cpu(), torch.min(img).cpu())
#     test_imgs.append(img)

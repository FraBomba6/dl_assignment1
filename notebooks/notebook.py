# %%
# Utils imports
import os
from tqdm import tqdm
from rich.console import Console

# Torch-related imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Custom imports
import libs.utils as custom_utils
import libs.net_utils as net_utils
from libs.loss import Loss
from libs.custom_dataset import CustomDataset

console = Console()
console.log("Initializing model parameters")
# Set torch seed
torch.manual_seed(3407)

# Initialize training variables
BATCH = 4
LR = 0.001
MOMENTUM = 0.9

# %%
# Loading training dataset
console.log("Building dataset")
train_dataset = CustomDataset(os.path.join(custom_utils.PROJECT_ROOT, "data", "assignment_1", "train"))

# Building training dataloader
console.log("Building dataloader")
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH,
    shuffle=True,
    collate_fn=custom_utils.collate_fn
)


# %%
class ObjectDetectionModel(nn.Module):
    def __init__(self, num_convolutions: int, out_filter: int, conv_k_sizes: list, pool_k_sizes: list):
        super(ObjectDetectionModel, self).__init__()
        if len(conv_k_sizes) != len(pool_k_sizes) or len(conv_k_sizes) != num_convolutions or len(pool_k_sizes) != num_convolutions:
            raise RuntimeError("Mismatch in length of arguments")
        in_filter = 3
        self.conv_blocks = nn.Sequential()
        for i in range(num_convolutions):
            block = net_utils.build_low_level_feat(in_filter, out_filter, conv_k_sizes[i], pool_k_sizes[i])
            self.conv_blocks.append(block)
            in_filter = out_filter
            out_filter *= 2
        self.inception1 = net_utils.build_inception_components(in_filter, out_filter)
        in_filter = out_filter * 6
        out_filter = in_filter * 2
        self.inception2 = net_utils.build_inception_components(in_filter, out_filter)
        self.batch_after_inception2 = nn.BatchNorm2d(out_filter*6)
        self.activation_after_inception = nn.ReLU()
        self.pool_after_inception = nn.MaxPool2d(2, 2)
        self.output = net_utils.build_output_components(out_filter*6)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = [
            self.inception1[0](x),
            self.inception1[1](x),
            self.inception1[2](x),
            self.inception1[3](x)
        ]
        x = torch.cat(x, 1)
        x = self.activation_after_inception(x)
        x = self.pool_after_inception(x)
        x = [
            self.inception2[0](x),
            self.inception2[1](x),
            self.inception2[2](x),
            self.inception2[3](x)
        ]
        x = torch.cat(x, 1)
        x = self.activation_after_inception(x)
        x = self.pool_after_inception(x)
        x = self.batch_after_inception2(x)
        x = [
            self.output[0](x),
            self.output[1](x),
            self.output[2](x)
        ]
        return torch.cat(x, 1)


# %%
def train(num_epochs):
    best_accuracy = 0.0

    network.to(custom_utils.DEVICE)

    for epoch in range(num_epochs):
        running_loss = 0.

        for i, data in enumerate(tqdm(train_dataloader)):
            images, target = data
            images = images.to(custom_utils.DEVICE)

            optimizer.zero_grad()

            outputs = network(images)

            loss_fn_return = loss_fn(outputs, target)
            loss = loss_fn_return[0]
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            bb = loss_fn_return[1][0]
            obj = loss_fn_return[1][1]
            no_obj = loss_fn_return[1][2]
            cla = loss_fn_return[1][3]
            if i % 10 == 9:
                print(
                    '[%d, %5d] loss: %.3f - bb: %.3f | obj: %.3f | no_obj: %.3f | class: %.3f' %
                    (epoch + 1, i + 1, running_loss / 10, bb, obj, no_obj, cla)
                )
                running_loss = 0.0


# %%
console.log("Creating model")
num_convolutions = 4
out_filter = 16
conv_k_sizes = [5, 3, 3, 3]
pool_k_sizes = [2, 2, 2, 1]
network = ObjectDetectionModel(num_convolutions, out_filter, conv_k_sizes, pool_k_sizes)

console.log("Initializing loss and optimizer")
loss_fn = Loss(5, 0.5)
optimizer = torch.optim.SGD(network.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=LR, max_lr=0.1)

console.log("Training")
train(25)

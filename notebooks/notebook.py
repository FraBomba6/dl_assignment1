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
LR = 0.01
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
    def __init__(self):
        super(ObjectDetectionModel, self).__init__()
        self.convolutions = nn.Sequential()
        # 256 x 256
        self.convolutions.append(net_utils.build_simple_convolutional_block(3, 16, conv_kernel=7, conv_stride=2))
        # 125 x 125
        self.convolutions.append(net_utils.build_simple_convolutional_block(16, 32, conv_kernel=5, pool_kernel=2))
        # 62 x 62
        self.convolutions.append(net_utils.build_simple_convolutional_block(32, 64, pool_kernel=2))
        # 31 x 31
        self.convolutions.append(net_utils.build_simple_convolutional_block(64, 128, pool_kernel=2))
        # 15 x 15
        self.convolutions.append(net_utils.build_simple_convolutional_block(128, 256, pool_kernel=2))
        # 7 x 7
        self.convolutions.append(net_utils.build_simple_convolutional_block(256, 256))
        self.convolutions.append(net_utils.build_simple_convolutional_block(256, 256))
        self.convolutions.append(net_utils.build_simple_convolutional_block(256, 128))
        self.output = net_utils.build_output_components(128)

    def forward(self, x):
        x = self.convolutions(x)
        x = [
            self.output[0](x),
            self.output[1](x),
            self.output[2](x)
        ]
        return torch.cat(x, 1)


# %%
def train(num_epochs, print_interval=10):
    best_accuracy = 0.0

    network.to(custom_utils.DEVICE)

    for epoch in range(num_epochs):
        running_loss = 0.
        epoch_loss = 0.

        for i, data in enumerate(tqdm(train_dataloader)):
            images, target = data
            images = images.to(custom_utils.DEVICE)

            outputs = network(images)
            optimizer.zero_grad()
            loss_fn_return = loss_fn(outputs, target)
            loss = loss_fn_return[0]
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item() * images.size(0)
            bb = loss_fn_return[1][0]
            obj = loss_fn_return[1][1]
            no_obj = loss_fn_return[1][2]
            cla = loss_fn_return[1][3]
            if i % print_interval == print_interval - 1:
                print(
                    '[%d, %5d] loss: %.3f - bb: %.3f | obj: %.3f | no_obj: %.3f | class: %.3f' %
                    (epoch + 1, i + 1, running_loss / print_interval, bb, obj, no_obj, cla)
                )
                running_loss = 0.0
        epoch_loss /= len(train_dataloader)
        console.log(":caution: Average epoch loss was %.3f for epoch %d" % (epoch_loss, epoch + 1))


# %%
console.log("Creating model")
network = ObjectDetectionModel()

console.log("Initializing loss and optimizer")
loss_fn = Loss(5, 0.5)
optimizer = torch.optim.Adam(network.parameters(), lr=LR)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=LR, max_lr=0.1)

console.log("Training")
train(25)

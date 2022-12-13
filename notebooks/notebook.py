# %%
# Utils imports
import os
from tqdm import tqdm
from rich.console import Console
from datetime import datetime

# Torch-related imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd

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
test_dataset = CustomDataset(os.path.join(custom_utils.PROJECT_ROOT, "data", "assignment_1", "test"))

# Building training dataloader
console.log("Building dataloader")
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH,
    shuffle=True,
    collate_fn=custom_utils.collate_fn
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
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
        self.convolutions.append(net_utils.build_simple_convolutional_block(3, 64, conv_kernel=7, conv_stride=2))
        self.convolutions.append(nn.ReLU())
        # 125 x 125
        self.convolutions.append(net_utils.build_simple_convolutional_block(64, 192, conv_kernel=3, pool_kernel=2))
        self.convolutions.append(net_utils.build_simple_convolutional_block(192, 128, conv_kernel=1))
        self.convolutions.append(nn.ReLU())
        # 62 x 62
        self.convolutions.append(net_utils.build_simple_convolutional_block(128, 256, conv_kernel=1))
        self.convolutions.append(net_utils.build_simple_convolutional_block(256, 256, pool_kernel=2))
        self.convolutions.append(nn.ReLU())
        self.convolutions.append(net_utils.build_simple_convolutional_block(256, 256, conv_kernel=1))
        self.convolutions.append(net_utils.build_simple_convolutional_block(256, 512, pool_kernel=2))
        self.convolutions.append(nn.ReLU())
        # 31 x 31
        for i in range(4):
            self.convolutions.append(net_utils.build_simple_convolutional_block(512, 256, conv_kernel=1))
            self.convolutions.append(net_utils.build_simple_convolutional_block(256, 512))
            self.convolutions.append(nn.ReLU())
        self.convolutions.append(net_utils.build_simple_convolutional_block(512, 512, conv_kernel=1))
        self.convolutions.append(net_utils.build_simple_convolutional_block(512, 1024, pool_kernel=2))
        self.convolutions.append(nn.ReLU())
        # 15 x 15
        for i in range(4):
            self.convolutions.append(net_utils.build_simple_convolutional_block(1024, 512, conv_kernel=1))
            self.convolutions.append(net_utils.build_simple_convolutional_block(512, 1024))
            self.convolutions.append(nn.ReLU())
        self.convolutions.append(net_utils.build_simple_convolutional_block(1024, 1024, pool_kernel=2))
        for i in range(3):
            self.convolutions.append(net_utils.build_simple_convolutional_block(1024, 1024))
            self.convolutions.append(nn.ReLU())
        # 7 x 7
        # self.output = net_utils.build_output_components(1024)
        self.output = nn.Sequential(
            nn.Linear(1024*7*7, 256*7*7),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(256*7*7, 64*7*7),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(64 * 7 * 7, 23 * 7 * 7)
        )

    def forward(self, x):
        x = self.convolutions(x)
        # x = [
        #     self.output[0](x),
        #     self.output[1](x),
        #     self.output[2](x)
        # ]
        # return torch.cat(x, 1)
        x = x.view(-1, 1024*7*7)
        x = self.output(x)
        return x.reshape(-1, 7, 7, 23)


# %%
def saveModel():
    path = os.path.join(custom_utils.PROJECT_ROOT, "models", datetime.now().strftime("%Y%m%d%H%M%S") + ".pth")
    torch.save(network.state_dict(), path)


def train(num_epochs, print_interval=10):
    best_accuracy = 0.0
    loss_data = {'epoch': [], 'loss': []}

    network.to(custom_utils.DEVICE)

    for epoch in range(num_epochs):
        network.train()
        console.log("\nTraining Epoch %d\n" % (epoch + 1))
        running_loss = 0.
        epoch_loss = 0.

        for i, data in enumerate(tqdm(train_dataloader)):
            images, target = data
            images = images.to(custom_utils.DEVICE)

            outputs = network(images)
            optimizer.zero_grad()
            loss_fn_return = loss_fn(outputs, target)
            loss = loss_fn_return[0]
            loss_data['epoch'].append(epoch)
            loss_data['loss'].append(loss.item())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item() * images.size(0)
            bb = loss_fn_return[1][0]
            obj = loss_fn_return[1][1]
            no_obj = loss_fn_return[1][2]
            cla = loss_fn_return[1][3]
            if i % print_interval == print_interval - 1:
                console.log(
                    '[%d, %5d] loss: %.3f - bb: %.3f | obj: %.3f | no_obj: %.3f | class: %.3f\n' %
                    (epoch + 1, i + 1, running_loss / print_interval, bb, obj, no_obj, cla)
                )
                running_loss = 0.0

        console.log("\nTesting accuracy\n")
        accuracy, accuracy_debug = test_accuracy()
        console.log('For epoch', epoch + 1, 'the test accuracy over the whole test set is %.2f%% [%.2f%%]' % (accuracy[0]*100, accuracy_debug[0]*100))

        # we want to save the model if the accuracy is the best
        if accuracy[0] > best_accuracy:
            saveModel()
            best_accuracy = accuracy[0]
    return pd.DataFrame.from_dict(loss_data)


def test_accuracy():
    network.eval()

    with torch.no_grad():
        batch_nms_boxes = []
        batch_targets = []

        for data_index, data in enumerate(tqdm(train_dataloader)):
            images, target = data
            images = images.to(custom_utils.DEVICE)

            outputs = network(images)

            for index, boxes in enumerate(outputs):
                boxes = custom_utils.from_prediction_to_box(boxes)
                nms_boxes = custom_utils.non_max_suppression(boxes)
                nms_boxes = [[data_index * BATCH + index] + box for box in nms_boxes]
                targets = target[6][index]
                targets = [[data_index * BATCH + index] + box for box in targets]
                for box in nms_boxes:
                    batch_nms_boxes.append(box)
                for box in targets:
                    batch_targets.append(box)

        console.log("The test accuracy over the train set is %.2f%%" % (custom_utils.mean_average_precision(batch_nms_boxes, batch_targets)[0].item()*100))

        for data_index, data in enumerate(tqdm(test_dataloader)):
            images, target = data
            images = images.to(custom_utils.DEVICE)

            outputs = network(images)

            for index, boxes in enumerate(outputs):
                boxes = custom_utils.from_prediction_to_box(boxes)
                nms_boxes = custom_utils.non_max_suppression(boxes)
                nms_boxes = [[data_index * BATCH + index] + box for box in nms_boxes]
                targets = target[6][index]
                targets = [[data_index * BATCH + index] + box for box in targets]
                for box in nms_boxes:
                    batch_nms_boxes.append(box)
                for box in targets:
                    batch_targets.append(box)

        return custom_utils.mean_average_precision(batch_nms_boxes, batch_targets), custom_utils.mean_average_precision(batch_targets, batch_targets)


# %%
console.log("Creating model")
network = ObjectDetectionModel()

console.log("Initializing loss and optimizer")
loss_fn = Loss(10, 0.5)
optimizer = torch.optim.Adam(network.parameters(), lr=LR)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=LR, max_lr=0.1)

loss_function_data = train(5, 500)
custom_utils.plot_loss(loss_function_data)

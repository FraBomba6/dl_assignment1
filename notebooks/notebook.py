# %%
# Utils imports
import json
import os
from tqdm import tqdm
from rich.console import Console

# Torch-related imports
import numpy as np
import math
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

# Custom imports
import libs.utils as custom_utils
import libs.net_utils as net_utils
from libs.loss import Loss

console = Console()
console.log("Initializing model parameters")
# Set torch seed
torch.manual_seed(3407)

# Initialize training variables
BATCH = 4
LR = 0.001
MOMENTUM = 0.9


# %%
class CustomDataset(Dataset):
    """
    Class that represents a dataset object to use as input on a CNN
    """
    def __init__(self, root):
        """
        Default initializer
        :param root: path to dataset root
        """
        self.root = root
        self.size = custom_utils.IMG_SIZE

        # Load images filelist
        self.images = list(sorted(os.listdir(os.path.join(root, "images"))))
        # Load annotations filelist
        self.annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def __getitem__(self, index):
        """
        Default getter for dataset objects
        :param index: i of the wanted image + annotation
        :return: image as PIL Image and target dictionary
        """
        img = self.__load_image(index)
        target = self.__generate_target(index)
        if self.size is not None:
            img, target = self.__apply_transform(img, target) 

        target["objectness"] = self.__compute_objectness(target['boxes'])
        target["boxes_mask"] = self.__build_target_bb_mask(target['boxes'])
        target["labels_mask"] = self.__build_target_labels_mask(target['objectness']['coords'], target['labels'])

        return img, target

    def __apply_transform(self, img, target):
        """
        Apply a resize transformation to an image and its target
        :param img: image as PIL Image
        :param target: dict representing the bounding boxes
        """
        target["boxes"] = self.__resize_boxes(target["boxes"], img.size)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((self.size, self.size))])
        img = transform(img)
        return img, target

    def __resize_boxes(self, boxes, img_size):
        """
        Apply to bounding boxes the same resize as the corresponding image
        :param boxes: tensor containing the coordinates of the bounding boxes
        :param img_size: size of the original image
        """
        x_scale = self.size/img_size[0]
        y_scale = self.size/img_size[1]

        scaled_boxes = []
        for box in boxes:
            box = box.tolist()
            x = int(np.round(box[0] * x_scale))
            y = int(np.round(box[1] * y_scale))
            x_max = int(np.round(box[2] * x_scale))
            y_max = int(np.round(box[3] * y_scale))
            scaled_boxes.append([x, y, x_max, y_max])
        return torch.as_tensor(scaled_boxes, dtype=torch.float32, device=custom_utils.DEVICE)

    def __load_image(self, index):
        """
        Load an image from the list of available images
        :param index: i of the wanted image
        :return: the image as a PIL.Image object
        """
        image_path = os.path.join(self.root, "images", self.images[index])
        return Image.open(image_path).convert("RGB")

    def __load_annotation(self, index):
        """
        Load image annotations from the list of available annotations files
        :param index: i of the wanted image
        :return: the annotations as a dict
        """
        annotation_path = os.path.join(self.root, "annotations", self.annotations[index])
        with open(annotation_path, "r") as fp:
            annotation_json = json.load(fp)
        return [value for key, value in annotation_json.items() if "item" in key]

    def __compute_objectness(self, boxes):
        target_matrix = np.zeros(49, dtype=np.float32).reshape(7, 7)
        coords = []
        square_length = np.round(self.size/7, 1)

        for box in boxes:
            box = box.tolist()
            box_center_x, box_center_y = np.round((box[2] - box[0]) / 2 + box[0], 1), np.round((box[3] - box[1]) / 2 + box[1], 1)
            box_center_x, box_center_y = math.floor(box_center_x / square_length), math.floor(box_center_y / square_length)
            target_matrix[box_center_y, box_center_x] = 1.0
            coords.append((box_center_x, box_center_y))

        return {"matrix": torch.as_tensor(target_matrix, dtype=torch.float32, device=custom_utils.DEVICE), "coords": coords}

    def __build_target_bb_mask(self, boxes):
        target_matrix = np.zeros(49*4, dtype=np.float32).reshape((7, 7, 4))
        square_length = np.round(self.size/7, 1)

        for box in boxes:
            box = box.tolist()
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]
            box_center_x = np.round(box[0] + box_w / 2, 1)
            box_center_y = np.round(box[1] + box_h / 2, 1)
            square_x, square_y = math.floor(box_center_x / square_length), math.floor(box_center_y / square_length)
            square_corner_x, square_corner_y = square_x * square_length, square_y * square_length
            box_center_x = (box_center_x - square_corner_x) / square_length
            box_center_y = (box_center_y - square_corner_y) / square_length
            box_w = box_w / self.size
            box_h = box_h / self.size
            target_matrix[square_y, square_x, 0] = box_center_x
            target_matrix[square_y, square_x, 1] = box_center_y
            target_matrix[square_y, square_x, 2] = box_w
            target_matrix[square_y, square_x, 3] = box_h
        return torch.as_tensor(target_matrix, dtype=torch.float32, device=custom_utils.DEVICE)

    def __build_target_labels_mask(self, coords, labels):
        target_matrix = np.zeros(49 * 13, dtype=np.float32).reshape((7, 7, 13))

        labels = labels.tolist()
        for index, label in enumerate(labels):
            target_matrix[coords[index][1], coords[index][0], label-1] = 1.

        return torch.as_tensor(target_matrix, dtype=torch.float32, device=custom_utils.DEVICE)

    def __generate_target(self, index):
        """
        Generate the target dict according to Torch specification
        :param index: i of the wanted annotations
        :return: target dict
        """
        annotations = self.__load_annotation(index)
        boxes = []
        labels = []
        categories = []
        
        for annotation in annotations:
            boxes.append(annotation["bounding_box"])
            labels.append(annotation["category_id"])
            categories.append(annotation['category_name'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32, device=custom_utils.DEVICE)
        labels = torch.as_tensor(labels, dtype=torch.int64, device=custom_utils.DEVICE)
        
        return {
            "boxes": boxes,
            "labels": labels,
            "categories": categories,
            "image_id": torch.tensor([index], device=custom_utils.DEVICE)
        }

    def __len__(self):
        return len(self.images)


# %%
# Loading training dataset
console.log("Building dataset")
train_dataset = CustomDataset(os.path.join(custom_utils.PROJECT_ROOT, "data", "assignment_1", "train"))

# Building training dataloader
console.log("Building dataloader")
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH, shuffle=True, collate_fn=custom_utils.collate_fn)


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

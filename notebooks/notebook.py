# %%
import json
import os
from random import randint
import libs.utils as custom_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import Dataset
from PIL import Image  # module
from PIL.Image import Image as PilImage  # object

# Defining project root in order to avoid relative paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initializing torch device according to hardware available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# determine the current device and based on that set the pin memory flag
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False


# %%
class CustomDataset(Dataset):
    """
    Class that represents a dataset object to use as input on a CNN
    """
    def __init__(self, root, size=None):
        """
        Default initializer
        :param root: path to dataset root
        :param size: optional target size for the image, if None no resizing
        """
        self.root = root
        self.size = size

        # Load images filelist
        self.images = list(sorted(os.listdir(os.path.join(root, "images"))))
        # Load annotations filelist
        self.annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def set_size(self, size):
        self.size = size

    def __getitem__(self, index):
        """
        Default getter for dataset objects
        :param index: index of the wanted image + annotation
        :return: image as PIL Image and target dictionary
        """
        img = self.__load_image(index)
        target = self.__generate_target(index)
        if self.size is not None:
            return self.__apply_transform(img, target)
        else:
            return img, target

    def __apply_transform(self, img, target):
        """
        Apply a resize transformation to an image and its target
        :param img: image as PIL Image
        :param target: dict representing the bounding boxes
        """
        target["boxes"] = self.__resize_boxes(target["boxes"], img.size)
        target["area"] = custom_utils.compute_area(target["boxes"])
        transform = transforms.Resize((self.size, self.size))
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
        return torch.as_tensor(scaled_boxes, dtype=torch.float32, device=DEVICE)

    def __load_image(self, index):
        """
        Load an image from the list of available images
        :param index: index of the wanted image
        :return: the image as a PIL.Image object
        """
        image_path = os.path.join(self.root, "images", self.images[index])
        return Image.open(image_path).convert("RGB")

    def __load_annotation(self, index):
        """
        Load image annotations from the list of available annotations files
        :param index: index of the wanted image
        :return: the annotations as a dict
        """
        annotation_path = os.path.join(self.root, "annotations", self.annotations[index])
        with open(annotation_path, "r") as fp:
            annotation_json = json.load(fp)
        return [value for key, value in annotation_json.items() if "item" in key]

    def __generate_target(self, index):
        """
        Generate the target dict according to Torch specification
        :param index: index of the wanted annotations
        :return: target dict
        """
        annotations = self.__load_annotation(index)
        boxes = []
        labels = []
        categories = []
        isCrowd = torch.zeros((len(annotations),), dtype=torch.int64)
        for annotation in annotations:
            boxes.append(annotation["bounding_box"])
            labels.append(annotation["category_id"])
            categories.append(annotation['category_name'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32, device=DEVICE)
        labels = torch.as_tensor(labels, dtype=torch.int64, device=DEVICE)
        area = custom_utils.compute_area(boxes)
        return {
            "boxes": boxes,
            "labels": labels,
            "categories": categories,
            "image_id": torch.tensor([index], device=DEVICE),
            "area": area,
            "isCrowd": isCrowd
        }

    def __len__(self):
        return len(self.images)


# %%
# Loading training dataset 

dataset = CustomDataset(os.path.join(PROJECT_ROOT, "data", "assignment_1", "train"))

# plot_size_distribution(dataset)

dataset.set_size(256)

# random image
image, target = dataset[randint(0, len(dataset))]

# check bounding box

custom_utils.with_bounding_box(image, target).show()


# %%
class ObjectDetectionModel(nn.Module):
    def __init__(self):
        super(ObjectDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.m_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.m_pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.m_pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.m_pool(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.m_pool(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


# %%

# %%
import json
import os
from random import randint
import libs.utils as custom_utils
import libs.net_utils as net_utils
import numpy as np
import math
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image  # module

# Set torch seed
torch.manual_seed(3407)

# Initialize training variables
BATCH = 16


# %%
class CustomDataset(Dataset):
    """
    Class that represents a dataset object to use as input on a CNN
    """
    def __init__(self, root):
        """
        Default initializer
        :param root: path to dataset root
        :param size: optional target size for the image, if None no resizing
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
        :param index: index of the wanted image + annotation
        :return: image as PIL Image and target dictionary
        """
        img = self.__load_image(index)
        target = self.__generate_target(index)
        if self.size is not None:
            img, target = self.__apply_transform(img, target) 

        target["objectness"] = self.__compute_objectness(target['boxes'])

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

    def __compute_objectness(self, boxes):
        target_matrix = np.zeros(49, dtype=np.float32).reshape(7, 7)
        coords = []

        for box in boxes:
            box = box.tolist()
            square_length = np.round(self.size/7, 1)
            box_centerx, box_centery = np.round((box[2] - box[0]) / 2 + box[0], 1), np.round((box[3] - box[1]) / 2 + box[1], 1)    
            box_centerx, box_centery = math.floor(box_centerx / square_length), math.floor(box_centery / square_length)
            target_matrix[box_centery, box_centerx] = 1.0
            coords.append((box_centerx, box_centery))

        return {"matrix": torch.as_tensor(target_matrix, dtype=torch.float32, device=custom_utils.DEVICE), "coords": coords}

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

train_dataset = CustomDataset(os.path.join(custom_utils.PROJECT_ROOT, "data", "assignment_1", "train"))

# plot_size_distribution(dataset)

# random image
image, target = train_dataset[randint(0, len(train_dataset))]
# transforms.ToPILImage()(target['objectness']["matrix"]).show()

print(target['objectness']["matrix"])

# check bounding box

# custom_utils.with_bounding_box(image, target).show()

# Building training dataloader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=0, collate_fn=custom_utils.collate_fn)


# %%
class ObjectDetectionModel(nn.Module):
    def __init__(self):
        super(ObjectDetectionModel, self).__init__()
        self.block1 = net_utils.build_low_level_feat(3, 32, 5, 4)
        self.block2 = net_utils.build_low_level_feat(32, 64, 3, 2)
        self.inception1 = net_utils.build_inception_components(64, 128)
        self.inception2 = net_utils.build_inception_components(128*6, 128*12)
        self.batch_after_inception2 = nn.BatchNorm2d(128*12*6)
        self.activation_after_inception = nn.ReLU()
        self.pool_after_inception = nn.MaxPool2d(2, 2)
        self.output = net_utils.build_output_components(128*12*6)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
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
        x = self.batch_after_inception2(x)
        x = self.activation_after_inception(x)
        x = self.pool_after_inception(x)
        x = [
            self.output[0](x),
            self.output[1](x),
            self.output[2](x)
        ]
        return torch.cat(x, 1)


# %%
network = ObjectDetectionModel()
network.to(custom_utils.DEVICE)

# %%
iterator = iter(train_dataloader)
images, boxes, labels, objectness_list = next(iterator)
images = images.to(custom_utils.DEVICE)

# %%
output = network(images)

# %%
p_boxes = []
p_labels = []
p_objectness = []
for img in output:
    p_boxes.append(img[1:5])
    p_labels.append(img[5:])
    p_objectness.append(img[0].reshape(49))
p_boxes = torch.stack(p_boxes)
p_labels = torch.stack(p_labels)
p_objectness = torch.stack(p_objectness)

objectness = torch.stack([entry['matrix'] for entry in objectness_list]).reshape(BATCH, 49)
cel = nn.CrossEntropyLoss()
cel_value = cel(p_objectness, objectness)

objects_coords = [entry['coords'] for entry in objectness_list]
p_batch_coords = []
for index, objects in enumerate(objects_coords):
    p_box = p_boxes[index]
    p_coords = []
    for box in objects:
        p_box_coords = []
        for filter in p_box:
            p_box_coords.append(filter[box[1]][box[0]])
        p_coords.append(p_box_coords)
    p_batch_coords.append(p_coords)


# %%
class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()

    def forward(self, outputs, boxes, labels, objectness):
        p_boxes = []
        p_labels = []
        p_objectness = []
        for img in outputs:
            p_boxes.append(img[1:5])
            p_labels.append(img[5:])
            p_objectness.append(img[0].reshape(49))
        p_boxes = torch.stack(p_boxes)
        p_labels = torch.stack(p_labels)
        p_objectness = torch.stack(p_objectness)

        objectness = torch.stack(objectness).reshape(BATCH, 49)
        cel = nn.CrossEntropyLoss()
        cel_value = cel(p_objectness, objectness)


# %%
# TODO Send network, image and target to device

# %%

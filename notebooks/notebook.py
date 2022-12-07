# %%
import json
import os
from random import randint
from tqdm import tqdm
import libs.utils as custom_utils
import libs.net_utils as net_utils
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as f
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image  # module
from rich.console import Console

console = Console()
console.log("Initializing model parameters")

# Set torch seed
torch.manual_seed(3407)

# Initialize training variables
BATCH = 16
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
        target_matrix = np.zeros(64, dtype=np.float32).reshape(8, 8)
        coords = []

        for box in boxes:
            box = box.tolist()
            square_length = np.round(self.size/8, 1)
            box_centerx, box_centery = np.round((box[2] - box[0]) / 2 + box[0], 1), np.round((box[3] - box[1]) / 2 + box[1], 1)    
            box_centerx, box_centery = math.floor(box_centerx / square_length), math.floor(box_centery / square_length)
            target_matrix[box_centery, box_centerx] = 1.0
            coords.append((box_centerx, box_centery))

        return {"matrix": torch.as_tensor(target_matrix, dtype=torch.float32, device=custom_utils.DEVICE), "coords": coords}

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

# %%

train_dataset = CustomDataset(os.path.join(custom_utils.PROJECT_ROOT, "data", "assignment_1", "train"))

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH,
    shuffle=True,
    collate_fn=custom_utils.collate_fn)

# plot_size_distribution(dataset)

# random image
# image, target = train_dataset[randint(0, len(train_dataset))]
# transforms.ToPILImage()(target['objectness']["matrix"]).show()

# print(target['objectness']["matrix"])

# check bounding box

# custom_utils.with_bounding_box(image, target).show()

# Building training dataloader


# %%
class ObjectDetectionModel(nn.Module):
    def __init__(self):
        super(ObjectDetectionModel, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding='same')
        self.pool = nn.MaxPool2d(kernel_size=2, stride = None)

        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding='same')

        self.prob = nn.Conv2d(32, 1, kernel_size=3, padding='same')
        self.boxes = nn.Conv2d(32, 4, kernel_size=3, padding='same')
        self.classes = nn.Conv2d(32, custom_utils.NUM_CATEGORIES, kernel_size=3, padding='same')

    def forward(self, x):

        # 128 x 128

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x) # 64 x 64
        x = self.norm1(x)


        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x) # 32 x 32
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x) # 16 x 16
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x) # 8 x 8 
        x = self.norm1(x)

        probability = self.prob(x)
        probability = self.sigmoid(probability)
        bounding_box = self.boxes(x) # linear regression
        classes = self.classes(x)
        classes = self.sigmoid(classes)

        
        # gate on the output in order not to backpropagate
        # on wrong outputs

        # gate = torch.where(probability > 0.5, torch.ones_like(probability), torch.zeros_like(probability))
        # bounding_box = bounding_box * gate
        # classes = classes * gate

        bounding_box = torch.where(bounding_box > 0.5, 1.0, 0.0)
        classes = torch.where(classes > 0.5, 1.0, 0.0)

        output = [
            probability,
            bounding_box,
            classes
        ]
        
        return torch.cat(output, 1)


class CustomLoss(nn.Module):

    def __init__(self,) -> None:
        super(CustomLoss, self).__init__()

    def __loss_bounding_box(self, y_true, y_pred):
        index = [i for i in range(1, 5)]
        y_true = torch.gather(y_true, index, axis=-1)
        y_pred = torch.gather(y_pred, index, axis=-1)

        loss = torch.nn.MSELoss(reduction='mean')
        return loss(y_pred, y_true)

    def __loss_probability(self, y_true, y_pred):
        index = [0]
        y_true = torch.gather(y_true, index, axis=-1)
        y_pred = torch.gather(y_pred, index, axis=-1)

        return f.binary_cross_entropy(y_pred, y_true, reduction='sum')

    def __loss_classes(self, y_true, y_pred):
        index = [i for i in range(5, 18)]
        y_true = torch.gather(y_true, index, axis=-1)
        y_pred = torch.gather(y_pred, index, axis=-1)

        return f.binary_cross_entropy(y_pred, y_true, reduction='sum')

    def forward(self, y_true, y_pred):
        return self.__loss_bounding_box(y_true, y_pred) + self.__loss_probability(y_true, y_pred) + self.__loss_classes(y_true, y_pred)
        
class YoloLoss(nn.Module):
    def __init__(self, l1, l2, l3):
        super(YoloLoss, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

    def forward(self, outputs, boxes, labels, objectness_list):
        current_batch_size = outputs.size()[0]
        # Set up predicted values
        p_boxes = []
        p_labels = []
        p_objectness = []
        for img in outputs:
            p_boxes.append(img[1:5])
            p_labels.append(img[5:])
            p_objectness.append(img[0].reshape(64))
        p_boxes = torch.stack(p_boxes)
        p_labels = torch.stack(p_labels)
        p_objectness = torch.stack(p_objectness)
        objects_coords = [entry['coords'] for entry in objectness_list]

        # Compute class loss
        cel_class_value = 0
        for i, objects in enumerate(objects_coords):
            p_label = p_labels[i]
            for j, box in enumerate(objects):
                p_label_values = []
                for filter in p_label:
                    p_label_values.append(filter[box[1]][box[0]])
                p_label_values = torch.tensor(p_label_values, dtype=torch.float32, device=custom_utils.DEVICE)
                cel_class_value += f.cross_entropy(p_label_values, labels[i][j] - 1)
        cel_class_value /= current_batch_size

        # Compute objectness loss
        objectness = torch.stack([entry['matrix'] for entry in objectness_list]).reshape(current_batch_size, 64)
        cel_obj_value = f.binary_cross_entropy(p_objectness, objectness)

        # Compute bb loss
        batch_bb_loss = 0
        for i, objects in enumerate(objects_coords):
            p_box = p_boxes[i]
            for j, box in enumerate(objects):
                p_box_coords = []
                for filter in p_box:
                    p_box_coords.append(filter[box[1]][box[0]].item())
                p_box_coords = torch.tensor(p_box_coords, dtype=torch.float32, device=custom_utils.DEVICE)
                target = torch.tensor(self.__compute_squared_error(boxes[i][j]), dtype=torch.float32, device=custom_utils.DEVICE)
                batch_bb_loss += f.mse_loss(p_box_coords, target)
        batch_bb_loss /= current_batch_size

        return self.l1 * cel_obj_value + self.l2 * batch_bb_loss + self.l3 * cel_class_value, (cel_obj_value, batch_bb_loss, cel_class_value)

    def __compute_squared_error(self, x_comp):
        x_comp = x_comp.cpu()

        # v2 is scaled from image size to 1
        scale = np.vectorize(lambda x: np.round(x / custom_utils.IMG_SIZE, 1))
        x_comp_scaled = scale(x_comp)

        return x_comp_scaled

# %%

console.log("Initializing model, loss and optimizer")
network = ObjectDetectionModel()
loss_fn = YoloLoss(1, 5, 5)
optimizer = torch.optim.SGD(network.parameters(), lr=LR, momentum=MOMENTUM)


# %%


def train(num_epochs):
    network.to(custom_utils.DEVICE)

    for epoch in range(num_epochs):
        running_loss = 0.

        for i, data in enumerate(tqdm(train_dataloader)):
            images, boxes, labels, objectness = data
            images = images.to(custom_utils.DEVICE)

            optimizer.zero_grad()

            outputs = network(images)

            loss_fn_return = loss_fn(outputs, boxes, labels, objectness)
            loss = loss_fn_return[0]
            loss.backward()

            optimizer.step()

            running_loss += loss.item()  # extract the loss value
            obj = loss_fn_return[1][0]
            bb = loss_fn_return[1][1]
            cla = loss_fn_return[1][2]
            if i % 10 == 9:
                # print every 1000 (twice per epoch)
                print(
                    '[%d, %5d] loss: %.3f - obj: %.3f | bb: %.3f | class: %.3f' %
                    (
                        epoch + 1,
                        i + 1,
                        running_loss / 10,
                        obj,
                        bb,
                        cla
                    )
                )
                # zero the loss
                running_loss = 0.0


# %%
console.log("Training")
train(3)

# %%

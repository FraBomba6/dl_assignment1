import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def collate_fn(batch):
    """
    Function to combine images, boxes and labels
    :param batch: an iterable of N sets from __getitem__() of CustomDataset
    :return: a tensor of images, lists of varying-size tensors of bounding boxes and labels
    """
    images = list()
    boxes = list()
    labels = list()
    objectness = list()

    for b in batch:
        images.append(b[0])
        boxes.append(b[1]["boxes"])
        labels.append(b[1]["labels"])
        objectness.append(b[1]["objectness"])

    images = torch.stack(images, dim=0)

    return images, boxes, labels, objectness


def with_bounding_box(image, target):
    """
    Returns an image with bounding boxes and labels
    :param image: image as Tensor
    :param target: dict representing containing the bounding boxes
    """
    tensor_image = torchvision.utils.draw_bounding_boxes(transforms.PILToTensor()(transforms.ToPILImage()(image)), target['boxes'], target['categories'], colors="red", width=2)
    return transforms.ToPILImage()(tensor_image)


def plot_aspect_ratio_distribution(dataset):
    """
    Returns the aspect ratio distribution of a CustomDataset
    :param dataset: the dataset of type CustomDataset
    """
    aspect_ratios = np.empty(len(dataset), dtype=float)
    for i in tqdm(range(len(dataset))):
        img, _ = dataset[i]
        sizes = img.size
        aspect_ratios = np.append(aspect_ratios, sizes[0] / sizes[1])

    plt.bar(*np.unique(aspect_ratios, return_counts=True))
    return plt

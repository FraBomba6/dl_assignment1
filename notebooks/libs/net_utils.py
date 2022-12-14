import torch
import torch.nn as nn
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_low_level_feat(in_channels, out_channels, conv_k_size, pool_k_size):
    """
    Builds a low level feature extraction block
    :param in_channels: input channels for the block
    :param out_channels: target output channels (there is no variation inside the block)
    :param conv_k_size: kernel size for convolution
    :param pool_k_size: kernel size for pooling | stride value
    :return Sequential object [Conv -> ReLU -> Conv -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool]
    """
    layers = nn.Sequential()
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=conv_k_size, padding=1))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=conv_k_size, padding=1))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=conv_k_size, padding=1))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=pool_k_size, stride=pool_k_size))
    return layers


def build_inception_components(in_channels, out_channels):
    """
    Builds the inception network components
    :param in_channels: input channels for the block
    :param out_channels: for the four components will be [out_channels, out_channels, out_channels*2, out_channels*2]
    :return the four components of an inception block
    """
    pool = nn.Sequential(
        nn.MaxPool2d(3, 1, padding=1),
        nn.Conv2d(in_channels, out_channels, kernel_size=1)
    ).to(DEVICE)
    conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1).to(DEVICE)
    conv2 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels * 2, kernel_size=1),
        nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=3, padding=1)
    ).to(DEVICE)
    conv3 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels * 2, kernel_size=1),
        nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=5, padding=2)
    ).to(DEVICE)
    return pool, conv1, conv2, conv3


def build_output_components(in_channels):
    """
    Builds the three output components of a YOLO-style network
    :param in_channels: input channels for the block
    :return the three components
    """
    confidence = nn.Sequential(
        nn.Conv2d(in_channels, 1, 1),
        nn.Sigmoid()
    ).to(DEVICE)
    box = nn.Sequential(
        nn.Conv2d(in_channels, 4, 1),
        nn.Conv2d(4, 4, 9, padding=4),
        nn.Conv2d(4, 4, 1),
        nn.ReLU()
    ).to(DEVICE)
    classes = nn.Sequential(
        nn.Conv2d(in_channels, 12, 1),
        nn.Softmax(dim=1)
    ).to(DEVICE)
    return confidence, box, classes

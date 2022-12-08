import torch
from torch import nn as nn
import utils as custom_utils


class Loss(nn.Module):
    def __init__(self, l1, l2):
        super(Loss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.l1 = l1
        self.l2 = l2

    def forward(self, predictions, targets):
        predictions = predictions.reshape(-1, 7, 7, 23)  # BATCH x 7 x 7 x 23 tensor output of the network
        target_boxes_mask = targets[4]  # 7 x 7 x 4 tensor containing coords of the center of the bb, width and height
        objectness_mask = targets[3].unsqueeze(3)  # 7 x 7 x 1 tensor containing 1 if an object is present

        iou_maxes, best_box = self.__find_best_bb(predictions[..., 2:10], target_boxes_mask)

        box_loss = self.__compute_box_loss(predictions, target_boxes_mask, objectness_mask, best_box)

        confidence_score = self.__compute_confidence_score(best_box, predictions)

        object_loss = self.__compute_object_loss(objectness_mask, confidence_score, targets[3])

        no_object_loss = self.__compute_no_object_loss(objectness_mask, predictions, targets[3])

        class_loss = self.__compute_class_loss(objectness_mask, predictions, targets[5])

        return self.l1 * box_loss + object_loss + self.l2 * no_object_loss + class_loss,\
            (self.l1 * box_loss, object_loss, self.l2 * no_object_loss, class_loss)

    def __compute_box_loss(self, predictions, target_boxes_mask, objectness_mask, best_box):
        """
        Compute the MSE on the bounding box predictions of the network
        :param predictions: BATCH x 7 x 7 x 23 tensor output of the network
        :param target_boxes_mask: BATCH x 7 x 7 x 4 tensor containing the coordinates of the target bounding boxes
        :param objectness_mask: BATCH x 7 x 7 x 1 tensor containing 1s where an object is present in the target
        :param best_box: BATCH x 7 x 7 x 1 tensor containing the index of which predicted bounding box to use
        :return: tensor containing the value of the loss
        """
        box_predictions = self.__compute_valid_boxes(predictions[..., 2:10], objectness_mask, best_box)

        box_targets = objectness_mask * target_boxes_mask

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        return self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

    def __compute_object_loss(self, objectness_mask, confidence_score, target):
        """
        Compute the MSE on the prediction of the presence of an object in a cell and the target
        :param objectness_mask: BATCH x 7 x 7 x 1 tensor containing 1s where an object is present in the target
        :param confidence_score: BATCH x 7 x 7 x 1 tensor containing the prediction of the presence of an object
        :param target: BATCH x 7 x 7 x 1 tensor ground truth of the presence of the object in the target
        :return: tensor containing the value of the loss
        """
        return self.mse(
            torch.flatten(objectness_mask * confidence_score),
            torch.flatten(target)
        )

    def __compute_no_object_loss(self, objectness_mask, predictions, targets):
        """
        Computes the MSE on the prediction of the absence of an object in a cell and the target
        :param objectness_mask: BATCH x 7 x 7 x 1 tensor containing 1s where an object is present in the target
        :param predictions: BATCH x 7 x 7 x 23 tensor containing all the predictions of the network
        :param targets: BATCH x 7 x 7 x 1 tensor ground truth of the presence of the object in the target
        :return: tensor containing the value of the loss
        """
        return self.mse(
            torch.flatten(((1 - objectness_mask) * predictions[..., 0:1]), start_dim=1),
            torch.flatten(((1 - objectness_mask) * targets.unsqueeze(3)), start_dim=1)
        ) + self.mse(
            torch.flatten(((1 - objectness_mask) * predictions[..., 1:2]), start_dim=1),
            torch.flatten(((1 - objectness_mask) * targets.unsqueeze(3)), start_dim=1)
        )

    def __compute_class_loss(self, objectness_mask, predictions, target):
        """
        Compute the MSE on the prediction of the class of an object and the target
        :param objectness_mask: BATCH x 7 x 7 x 1 tensor containing 1s where an object is present in the target
        :param predictions: BATCH x 7 x 7 x 23 tensor containing all the predictions of the network
        :param target: BATCH x 7 x 7 x 13 tensor ground truth of the class of an object
        :return:
        """
        return self.mse(
            torch.flatten(objectness_mask * predictions[..., 10:], end_dim=-2),
            torch.flatten(objectness_mask * target, end_dim=-2)
        )

    def __compute_confidence_score(self, best_box, predictions):
        """
        Compute confidence score according to YOLOv1
        """
        return best_box * predictions[..., 1:2] + (1 - best_box) * predictions[..., 0:1]

    def __compute_valid_boxes(self, predictions, objectness_mask, best_box):
        """
        Computes the valid predictions based on best bounding box and valid objectness
        """
        return objectness_mask * (best_box * predictions[..., 4:8] + (1 - best_box) * predictions[..., 0:4])

    def __find_best_bb(self, predictions, target_boxes_mask):
        """
        Computes the best predicted bounding box using Intersection Over Union
        """
        # Computes IOU on first predicted bounding box and target
        iou_bb_1 = custom_utils.i_over_u(predictions[..., 0:4], target_boxes_mask)
        # Computes IOU on first predicted bounding box and target
        iou_bb_2 = custom_utils.i_over_u(predictions[..., 4:8], target_boxes_mask)
        # Merge the previous two into a (2, BATCH, 7, 7, 4) Tensor
        iou_bbs = torch.cat([iou_bb_1.unsqueeze(0), iou_bb_2.unsqueeze(0)], dim=0)
        return torch.max(iou_bbs, dim=0)  # Return best bounding box for each mask cell (maximum IOU)

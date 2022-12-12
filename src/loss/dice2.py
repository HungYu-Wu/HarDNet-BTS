import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class EDiceLoss2(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=True):
        super(EDiceLoss2, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["ET", "TC", "WT"]
        self.device = "cpu"

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.
        
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)
        
        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:
                    return torch.tensor(1., device="cuda")
                else:
                    return torch.tensor(0., device="cuda")
            # Threshold the pred
        
        intersection = EDiceLoss2.compute_intersection(inputs, targets)
        
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            targets = targets.float()
            inputs1 = inputs
            inputs1 = inputs1.float()
            wbce  = EDiceLoss2.ce_loss(inputs1, targets,1)
            dice = (2 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
            dice = 1-dice
        if metric_mode:
            return dice
        
        return 0.2*wbce + 0.8*dice
        
        
    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target):
        dice = 0
        
        for i in range(target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)

        final_dice = dice / target.size(1)
        return final_dice

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))
            dices.append(dice)
        return dices
    def ce_loss(true, logits, weights, ignore=255):
        """Computes the weighted multi-class cross-entropy loss.
        Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        weight: a tensor of shape [C,]. The weights attributed
            to each class.
        ignore: the class index to ignore.
        Returns:
        ce_loss: the weighted multi-class cross-entropy loss.
        """
        L = nn.BCEWithLogitsLoss()
        ce_loss = L(true,logits)
        return ce_loss



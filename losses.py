import torch
from torch import nn


class MaskedMultiTaskCrossEntropy(nn.Module):

    def forward(self, input, target):
        scores = torch.sigmoid(input)
        target_active = (target == 1).float()  # from -1/1 to 0/1
        loss_terms = -(target_active * torch.log(scores) + (1 - target_active) * torch.log(1 - scores))
        missing_values_mask = (target != 0).float()
        return (loss_terms * missing_values_mask).sum() / missing_values_mask.sum()


LOSS_FUNCTIONS = {
    'MaskedMultiTaskCrossEntropy': MaskedMultiTaskCrossEntropy(),
    'MSE': nn.MSELoss()
}



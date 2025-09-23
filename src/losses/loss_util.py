import torch

CLASS_WTS = torch.tensor([2.25 , 1.0 , 3.75 , 2.5 , 1.65 , 1.75], dtype=torch.float)

class FocalLoss(torch.nn.Module):
    '''
    Focal Loss for multi-class classification
    Ref: Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(2), 318–327. https://doi.org/10.1109/TPAMI.2018.2858826
    Usage:
        criterion = FocalLoss(alpha=torch.tensor([1.0, 2.0, 3.0]), gamma=2)
        loss = criterion(inputs, targets)
    where alpha is a tensor of shape (num_classes,) representing the weight for each class, and gamma is a focusing parameter.
    Note: Make sure to adjust the alpha values according to your class distribution.
    '''
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # This project aims to classify 5 classes of ovarian conditions, so use a slightly larger label_smoothing
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none', label_smoothing=0.1)
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss
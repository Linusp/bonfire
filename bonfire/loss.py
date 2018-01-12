import torch
from torch.nn import Module
from torch.autograd import Variable
from torch.nn.functional import log_softmax


class FocalLoss(Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, inputs, target):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))  # N,H*W,C => N*H*W,C

        target = target.view(-1, 1)

        logpt = log_softmax(inputs, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def pullaway_loss(embeddings):
    norm = torch.sqrt(torch.sum(embeddings ** 2.0, 1, keepdim=True))
    normalized_embeddings = embeddings / norm
    similarity = torch.matmul(normalized_embeddings, normalized_embeddings.transpose(1, 0))
    batch_size = embeddings.size()[0]
    pt_loss = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
    return pt_loss

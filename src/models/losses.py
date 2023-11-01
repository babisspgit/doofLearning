import torch
import torch.nn as nn
import numpy as np


class ConstrastiveLoss(nn.Module):
    def __init__(self, m=1.0):
        super().__init__()

        self.m = m

    def forward(
        self, item1: torch.Tensor, item2: torch.Tensor, similar: bool = True
    ) -> torch.float32:
        # if flag == 1 or true : Similar items
        # if flag == 0 or false : Dissimilar items

        L2_dist = torch.pairwise_distance(item1, item2)

        # Since this is a loss function, I have to average over the batch size
        if similar:
            return torch.mean(torch.pow(L2_dist, 2))

        result = self.m - L2_dist

        return torch.mean(torch.clamp(result, min=0.0))


class ClipSymmetricLoss(nn.Module):
    """
    From the paper
    Learning Transferable Visual Models From Natural Language Supervision
    https://arxiv.org/abs/2103.00020
    """

    def __init__(self, t: float):
        super().__init__()
        self.t = t  # temperature

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, img_emb, text_emb):
        p = 2  # for l2 normalization
        img_emb_norm = torch.nn.functional.normalize(img_emb, p, dim=1)
        text_emb_norm = torch.nn.functional.normalize(text_emb, p, dim=1)

        # scaled pairwise cosine similarities [n, n]
        logits = torch.matmul(img_emb_norm, text_emb_norm.transpose(1, 0)) * np.exp(
            self.t
        )
        logits = logits.to(img_emb.device)

        # symmetric loss function
        labels = torch.arange(logits.shape[0])
        loss_i = self.cross_entropy_loss(logits, labels, axis=0)
        loss_t = self.cross_entropy_loss(logits, labels, axis=1)
        return (loss_i + loss_t) / 2

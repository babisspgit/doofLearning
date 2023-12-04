import torch
import torch.nn as nn
import numpy as np


def triplet_loss(
    anchor, positive, negative, margin=1.0, p=2, eps=1e-6, norm_embeddings=True
):
    """
    Triplet loss function.
    Args:
        anchor: anchor feature vector
        positive: positive feature vector
        negative: negative feature vector
        margin: margin for triplet loss
        p: norm degree
        eps: epsilon for numerical stability
    Returns:
        triplet loss value
    """
    assert anchor.size() == positive.size() and anchor.size() == negative.size()

    if norm_embeddings:
        anchor = anchor / anchor.norm(p=2, dim=-1, keepdim=True)
        positive = positive / positive.norm(p=2, dim=-1, keepdim=True)
        negative = negative / negative.norm(p=2, dim=-1, keepdim=True)

    dist_ap = (anchor - positive).norm(p=p, dim=1)

    dist_an = (anchor - negative).norm(p=p, dim=1)

    loss = torch.clamp(dist_ap - dist_an + margin, min=eps)

    return loss.mean()


class ConstrastiveLoss(nn.Module):
    def __init__(self, m=1.0):
        super().__init__()

        self.m = m

    def forward(
        self, item1: torch.Tensor, item2: torch.Tensor, similar: bool = True
    ) -> torch.float32:
        # if similar is true : Similar items
        # if similar is false : Dissimilar items

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
        # Cross_Ent : Applying a softmax function to the logits and then computing the negative log likelihood of the correct classes (which is what cross-entropy measures).
        loss_i = self.cross_entropy_loss(logits, labels)  # for image to text (rows)
        loss_t = self.cross_entropy_loss(logits, labels)  # for text to image (columns)
        return (loss_i + loss_t) / 2  # Average of the two losses

"""Texture-affinity losses used to train CoFTransTANet."""

import torch
import torch.nn.functional as F


def affinity(features, mask, affinity_sizes):
    return sum(affinity_with_size(features, mask, size) for size in affinity_sizes)


def affinity_with_size(features, mask, size):
    n, c, h, w = features.size()
    mask_height = mask.size(2)
    assert h == w
    assert h >= size
    assert h % size == 0

    features = F.avg_pool2d(features, kernel_size=h // size)
    h, w = features.size()[2:]
    features = features.view(n, c, h * w).transpose(1, 2)
    feature_norm = torch.norm(features, dim=2, keepdim=True)
    normalizer = torch.bmm(feature_norm, feature_norm.transpose(2, 1))
    feature_similarity = torch.bmm(features, features.transpose(2, 1)) / (normalizer + 1e-6)

    mask = F.avg_pool2d(mask, kernel_size=mask_height // size)
    mask = mask.view(n, 1, h * w).transpose(1, 2)
    mask_similarity = 2 * (0.5 - torch.abs(mask - mask.transpose(2, 1)))
    mask_similarity = (mask_similarity + 1) / 2

    positive_ratio = torch.sum(mask, dim=1, keepdim=True) / (h * w)
    spatial_weight = (1 - mask) * positive_ratio + mask * (1 - positive_ratio)
    spatial_weight = spatial_weight / (torch.sum(spatial_weight, dim=1, keepdim=True) + 1e-8)
    loss = F.l1_loss(feature_similarity, mask_similarity, reduction='none')
    loss = torch.sum(loss * spatial_weight, dim=1)
    loss = torch.sum(loss * spatial_weight.view(n, h * w), dim=1)
    return torch.mean(loss)

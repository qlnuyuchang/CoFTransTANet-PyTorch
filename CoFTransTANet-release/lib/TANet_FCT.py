import sys

from torch import nn
import torch.nn.functional as F
import torch
from torchvision import models
from lib.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.FCT import Block_encoder_bottleneck


# from feat_vis import draw_feature


class CoFTransTANet(nn.Module):
    def __init__(self, nf=32, imagenet_pretrained=True):
        super(CoFTransTANet, self).__init__()
        extractor = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)
        # extractor = models.resnet50(pretrained=True)

        self.layer0 = nn.Sequential(extractor.conv1, extractor.bn1, extractor.relu, extractor.maxpool)
        self.layer1 = extractor.layer1
        self.layer2 = extractor.layer2
        self.layer3 = extractor.layer3
        self.layer4 = extractor.layer4

        self.down1 = nn.Sequential(nn.Conv2d(256, nf, 1, bias=False), nn.BatchNorm2d(nf))
        self.down2 = nn.Sequential(nn.Conv2d(512, nf, 1, bias=False), nn.BatchNorm2d(nf))
        self.down3 = nn.Sequential(nn.Conv2d(1024, nf, 1, bias=False), nn.BatchNorm2d(nf))
        self.down4 = nn.Sequential(nn.Conv2d(2048, nf, 1, bias=False), nn.BatchNorm2d(nf))

        self.tarm1 = TARM(nf, nf)
        self.tarm2 = TARM(nf, nf)
        self.tarm3 = TARM(nf, nf)

        self.refine1 = nn.Sequential(
            nn.Conv2d(nf * 2, nf, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(nf), nn.ReLU(),
            nn.Conv2d(nf, nf, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(nf)
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(nf * 2, nf, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(nf), nn.ReLU(),
            nn.Conv2d(nf, nf, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(nf)
        )
        self.refine3 = nn.Sequential(
            nn.Conv2d(nf * 2, nf, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(nf), nn.ReLU(),
            nn.Conv2d(nf, nf, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(nf)
        )

        self.predict = nn.Sequential(
            nn.Conv2d(nf, nf // 4, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(nf // 4), nn.ReLU(),
            nn.Conv2d(nf // 4, 1, kernel_size=3, padding=1)
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        f0 = self.layer0(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        down1 = self.down1(f1)
        down2 = self.down2(f2)
        down3 = self.down3(f3)
        down4 = self.down4(f4)

        fea4 = down4
        fea4out = fea4

        fea3 = F.upsample(fea4out, size=down3.size()[2:], mode='bilinear', align_corners=True)
        fea3 = self.refine3(torch.cat((fea3, down3), 1)) + fea3
        fea3out, mean3, std3 = self.tarm3(fea3)

        fea2 = F.upsample(fea3out, size=down2.size()[2:], mode='bilinear', align_corners=True)
        fea2 = self.refine2(torch.cat((fea2, down2), 1)) + fea2
        fea2out, mean2, std2 = self.tarm2(fea2)

        fea = F.upsample(fea2out, size=down1.size()[2:], mode='bilinear', align_corners=True)
        fea1 = self.refine1(torch.cat((fea, down1), 1)) + fea
        fea1out, mean1, std1 = self.tarm1(fea1)

        fea2out = F.upsample(fea2out, size=fea1.size()[2:], mode='bilinear', align_corners=True)
        fea3out = F.upsample(fea3out, size=fea1.size()[2:], mode='bilinear', align_corners=True)
        fea4out = F.upsample(fea4out, size=fea1.size()[2:], mode='bilinear', align_corners=True)

        pre4 = F.upsample(self.predict(fea4out), x.size()[2:], mode='bilinear', align_corners=True)
        pre3 = F.upsample(self.predict(fea3out), x.size()[2:], mode='bilinear', align_corners=True)
        pre2 = F.upsample(self.predict(fea2out), x.size()[2:], mode='bilinear', align_corners=True)
        pre1 = F.upsample(self.predict(fea1out), x.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return pre4, pre3, pre2, pre1, mean1, mean2, mean3, std1, std2, std3
        # if self.training:
        #     return pre4, pre3, pre2, pre1
        # print(save_path)
        # draw_feature(fea, save_path=save_path)
        # draw_feature(fea1out, save_path=save_path1)
        #
        # sys.exit()
        #
        # draw_feature(fea1out, save_path=save_path1)

        return torch.sigmoid(pre1)


class TARM(nn.Module):
    def __init__(self, nf_in, nf_out):
        super(TARM, self).__init__()
        self.base_conv = nn.Sequential(nn.Conv2d(nf_in, nf_out, 3, 1, 1, bias=False))
        self.basic_conv = nn.ModuleList([nn.Sequential(nn.Conv2d(nf_in, 12, 1)) for _ in range(8)])
        self.fct_block = Block_encoder_bottleneck("first", 12, 78, 2)

        self.down = nn.Sequential(nn.Conv2d(624, nf_out, 1), nn.ReLU())
        self.std_conv = nn.Sequential(nn.Conv2d(nf_out, nf_out, 3, 1, 1))
        self.mean_conv = nn.Sequential(nn.Conv2d(nf_out, nf_out, 3, 1, 1))
        self.fuse = nn.Sequential(nn.Conv2d(nf_out, nf_out, 3, 1, 1), nn.BatchNorm2d(nf_out))
        # ind = torch.triu_indices(12, 12)
        # self.ind = torch.Tensor([ind[0][i] * 12 + ind[1][i] for i in range(len(ind[0]))]).long().cuda()

    def forward(self, x):
        infea = self.base_conv(x)
        basics = []
        for conv in self.basic_conv:
            temp = conv(x)
            # print(temp.size)
            temp = self.fct_block(temp)
            # print('shape:', temp.shape)
            # temp = F.upsample(temp, scale_factor=2, mode='bilinear', align_corners=True)
            # print(temp.shape)
            basics.append(temp)
        b = torch.cat(basics, dim=1)
        # print(b.shape)
        base = self.down(b)
        base = F.upsample(base, scale_factor=2, mode='bilinear', align_corners=True)
        # print(base.shape)
        std = self.std_conv(base)
        # print(std.shape)
        mean = self.mean_conv(base)
        batch_mean = torch.mean(infea, dim=(0, 2, 3), keepdim=True)
        batch_std = torch.std(infea, dim=(0, 2, 3), keepdim=True)
        infea = (infea - batch_mean) / (1e-8 + batch_std)

        return self.fuse(F.relu(infea * std + mean)) + x, mean, std


def affinity(assb_weight, mask, affinity_size):
    return sum([affinity_with_size(assb_weight, mask, size) for size in affinity_size])


def affinity_with_size(x, mask, size):
    n, c, h, w = x.size()
    m_h = mask.size(2)
    assert h == w
    assert h >= size
    assert h % size == 0

    x = F.avg_pool2d(x, kernel_size=h // size)
    h, w = x.size()[2:]
    x = x.view(n, c, h * w).transpose(1, 2)
    x_norm = torch.norm(x, dim=2, keepdim=True)
    temp = torch.bmm(x_norm, x_norm.transpose(2, 1))
    x_sim_mat = torch.bmm(x, x.transpose(2, 1)) / (temp + 1e-6)

    mask = F.avg_pool2d(mask, kernel_size=m_h // size)
    mask = mask.view(n, 1, h * w).transpose(1, 2)
    mask_sim_mat = 2 * (0.5 - torch.abs(mask - mask.transpose(2, 1)))
    mask_sim_mat = (mask_sim_mat + 1) / 2

    assert mask_sim_mat.size() == x_sim_mat.size()

    positive_ratio = torch.sum(mask, dim=1, keepdim=True) / (h * w)
    spatial_weight = (1 - mask) * positive_ratio + mask * (1 - positive_ratio)
    spatial_weight = spatial_weight / torch.sum(spatial_weight, dim=1, keepdim=True)
    loss = F.l1_loss(x_sim_mat, mask_sim_mat, reduction='none')
    loss = torch.sum(loss * spatial_weight, dim=1)
    loss = torch.sum(loss * spatial_weight.view(n, h * w), dim=1)
    loss = torch.mean(loss)
    return loss


def edge(feature, mask, affinity_size):
    return sum([edge_with_size(feature, mask, size) for size in affinity_size])


def edge_with_size(x, mask, size):
    n, c, h, w = x.size()
    m_h = mask.size(2)

    assert h == w
    assert h > size
    assert h % size == 0

    x = x.view(n, c, h // size, size, w // size, size).transpose(3, 4)
    x = x.contiguous().view(n, c, (h // size) * (w // size), size * size).transpose(1, 2)
    x = x.contiguous().view(n * (h // size) * (w // size), c, size * size).transpose(1, 2)
    x_norm = torch.norm(x, dim=2, keepdim=True)
    temp = torch.matmul(x_norm, x_norm.transpose(1, 2))
    temp = torch.where(temp == 0, torch.ones_like(temp), temp)
    x_sim_mat = torch.matmul(x, x.transpose(1, 2)) / temp

    mask0 = F.avg_pool2d(mask, kernel_size=m_h // h)
    mask = mask0.view(n, 1, h // size, size, w // size, size).transpose(3, 4)
    mask = mask.contiguous().view(n, 1, (h // size) * (w // size), size * size).transpose(1, 2)
    is_edge = torch.sum(mask, dim=3).view(n * (h // size) * (w // size), 1)
    mask = mask.contiguous().view(n * (h // size) * (w // size), 1, size * size).transpose(1, 2)
    mask_sim_mat = 2 * (0.5 - torch.abs(mask - mask.transpose(2, 1)))
    mask_sim_mat = (mask_sim_mat + 1) / 2

    assert mask_sim_mat.size() == x_sim_mat.size()
    loss = F.l1_loss(x_sim_mat, mask_sim_mat, reduction='none')
    loss = torch.sum(loss.view(n * (h // size) * (w // size), -1), dim=1, keepdim=True)
    loss = torch.where(is_edge == 0, torch.zeros_like(loss), loss)
    loss = torch.where(is_edge == size * size, torch.zeros_like(loss), loss)
    window_weight = torch.where(is_edge == 0, torch.zeros_like(is_edge),
                                torch.ones_like(is_edge) * size * size * size * size)
    window_weight = torch.where(is_edge == size * size, torch.zeros_like(window_weight),
                                torch.ones_like(window_weight) * size * size * size * size)

    return torch.sum(loss) / (1e-6 + torch.sum(window_weight))


# Backward-compatible alias used by the original training and testing scripts.
TANet = CoFTransTANet

# net = models.resnet50()
# layer0 = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
#
# print(net)

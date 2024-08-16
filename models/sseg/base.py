from models.backbone.hrnet import HRNet
from models.backbone.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d
import torchvision.transforms as T
import torch
from torch import nn
import torch.nn.functional as F
from ..utae import  UTAE
from ..src.fusion_utils import *
def get_backbone(backbone, pretrained):
    if backbone == "resnet18":
        backbone = resnet18(pretrained)
    elif backbone == "resnet34":
        backbone = resnet34(pretrained)
    elif backbone == "resnet50":
        backbone = resnet50(pretrained)
    elif backbone == "resnet101":
        backbone = resnet101(pretrained)
    elif backbone == "resnet152":
        backbone = resnet152(pretrained)

    elif backbone == "resnext50":
        backbone = resnext50_32x4d(pretrained)
    elif backbone == "resnext101":
        backbone = resnext101_32x8d(pretrained)

    elif "hrnet" in backbone:
        backbone = HRNet(backbone, pretrained)

    else:
        exit("\nError: BACKBONE \'%s\' is not implemented!\n" % backbone)

    return backbone


class BaseNet(nn.Module):
    def __init__(self, backbone, pretrained):
        super(BaseNet, self).__init__()
        self.backbone = get_backbone(backbone, pretrained)
        self.fusion = UTAE(input_dim=15)
        self.fm_utae_featmap_cropped = FM_cropped(32,
                                                  list([32, 32, 64, 270]),
                                                  )
        self.fm_utae_featmap_collapsed = FM_collapsed(32,
                                                      list([32, 32, 64, 270]),)

    def base_forward(self, x1, x2,utae_fmaps_dec):
        b, c, h, w = x1.shape

        unet_fmaps_enc = self.backbone.base_forward(x1)
        x1=unet_fmaps_enc[-1]
        x2 = self.backbone.base_forward(x2)[-1]

        transform = T.CenterCrop((26, 26))
        utae_last_fmaps_reshape_cropped = transform(utae_fmaps_dec[-1])
        utae_last_fmaps_reshape_cropped = self.fm_utae_featmap_cropped(utae_last_fmaps_reshape_cropped,
                                                                       [i.size()[-1] for i in utae_fmaps_dec])

        ### collapsed fusion module
        utae_fmaps_dec_squeezed = torch.mean(utae_fmaps_dec[-1][0], dim=(-2, -1))
        utae_last_fmaps_reshape_collapsed = self.fm_utae_featmap_collapsed(utae_fmaps_dec_squeezed,
                                                                           [i.size()[-1] for i in
                                                                            utae_fmaps_dec])  ### reshape last feature map of utae to match feature maps enc. unet

        ### adding cropped/collasped
        utae_last_fmaps_reshape = [torch.add(i, j) for i, j in
                                   zip(utae_last_fmaps_reshape_cropped, utae_last_fmaps_reshape_collapsed)]


        unet_utae_fmaps = utae_last_fmaps_reshape[-1]
        x1=x1+unet_utae_fmaps
        x2=x2+unet_utae_fmaps

        out1 = self.head(x1)
        out2 = self.head(x2)
        out3= self.head(unet_utae_fmaps)
        out1 = F.interpolate(out1, size=(h, w), mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, size=(h, w), mode='bilinear', align_corners=False)
        out3 = F.interpolate(out3, size=(h, w), mode='bilinear', align_corners=False)

        out_bin = torch.abs(x1 - x2)
        out_bin = self.head_bin(out_bin)
        out_bin = F.interpolate(out_bin, size=(h, w), mode='bilinear', align_corners=False)
        out_bin = torch.sigmoid(out_bin)

        return out1, out2, out_bin.squeeze(1),out3

    def forward(self, x1, x2,sstime_list, tta=False):
        if not tta:
            sstime_list2 = sstime_list
            time_out,utae_fmaps_dec=self.fusion(sstime_list2)
            return self.base_forward(x1, x2,utae_fmaps_dec)
        else:
            sstime_list3 = sstime_list
            time_out, utae_fmaps_dec = self.fusion(sstime_list3)

            out1, out2, out_bin ,out3= self.base_forward(x1, x2,utae_fmaps_dec)
            out1 = F.softmax(out1, dim=1)
            out2 = F.softmax(out2, dim=1)
            out_bin = out_bin.unsqueeze(1)
            origin_x1 = x1.clone()
            origin_x2 = x2.clone()

            x1 = origin_x1.flip(2)
            x2 = origin_x2.flip(2)
            cur_out1, cur_out2, cur_out_bin ,out3 = self.base_forward(x1, x2,utae_fmaps_dec)
            out1 += F.softmax(cur_out1, dim=1).flip(2)
            out2 += F.softmax(cur_out2, dim=1).flip(2)
            out_bin += cur_out_bin.unsqueeze(1).flip(2)

            x1 = origin_x1.flip(3)
            x2 = origin_x2.flip(3)
            cur_out1, cur_out2, cur_out_bin ,out3 = self.base_forward(x1, x2,utae_fmaps_dec)
            out1 += F.softmax(cur_out1, dim=1).flip(3)
            out2 += F.softmax(cur_out2, dim=1).flip(3)
            out_bin += cur_out_bin.unsqueeze(1).flip(3)

            x1 = origin_x1.transpose(2, 3).flip(3)
            x2 = origin_x2.transpose(2, 3).flip(3)
            cur_out1, cur_out2, cur_out_bin ,out3 = self.base_forward(x1, x2,utae_fmaps_dec)
            out1 += F.softmax(cur_out1, dim=1).flip(3).transpose(2, 3)
            out2 += F.softmax(cur_out2, dim=1).flip(3).transpose(2, 3)
            out_bin += cur_out_bin.unsqueeze(1).flip(3).transpose(2, 3)

            x1 = origin_x1.flip(3).transpose(2, 3)
            x2 = origin_x2.flip(3).transpose(2, 3)
            cur_out1, cur_out2, cur_out_bin ,out3 = self.base_forward(x1, x2,utae_fmaps_dec)
            out1 += F.softmax(cur_out1, dim=1).transpose(2, 3).flip(3)
            out2 += F.softmax(cur_out2, dim=1).transpose(2, 3).flip(3)
            out_bin += cur_out_bin.unsqueeze(1).transpose(2, 3).flip(3)

            x1 = origin_x1.flip(2).flip(3)
            x2 = origin_x2.flip(2).flip(3)
            cur_out1, cur_out2, cur_out_bin ,out3 = self.base_forward(x1, x2,utae_fmaps_dec)
            out1 += F.softmax(cur_out1, dim=1).flip(3).flip(2)
            out2 += F.softmax(cur_out2, dim=1).flip(3).flip(2)
            out_bin += cur_out_bin.unsqueeze(1).flip(3).flip(2)

            out1 /= 6.0
            out2 /= 6.0
            out_bin /= 6.0

            return out1, out2, out_bin.squeeze(1)

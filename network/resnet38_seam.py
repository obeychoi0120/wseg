import torch
import torch.nn as nn
import torch.nn.functional as F

import network.resnet38d


class Net(network.resnet38d.Net):
    def __init__(self, num_class=21):
        super().__init__()
        self.dropout7 = torch.nn.Dropout2d(0.5)
        self.fc8 = nn.Conv2d(4096, num_class, 1, bias=False)
        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        self.f9 = torch.nn.Conv2d(192 + 3, 192, 1, bias=False)

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)

        # Added: multi-layer inference
        # self.proj8_3 = torch.nn.Conv2d(64, 21, 1, bias=False)
        # self.proj8_4 = torch.nn.Conv2d(128, 21, 1, bias=False)
        # torch.nn.init.xavier_uniform_(self.proj8_3.weight)
        # torch.nn.init.xavier_uniform_(self.proj8_4.weight)

        self.from_scratch_layers = [self.f8_3, self.f8_4, self.f9, self.fc8] #self.proj8_3, self.proj8_4]
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

    def forward(self, img, return_feat=True):
        N, C, H, W = img.size()
        d = super().forward_as_dict(img)
        feat = d['conv6']
        cam = self.fc8(self.dropout7(feat))
        n,c,h,w = cam.size()

        with torch.no_grad():
            cam_d = F.relu(cam.detach())
            cam_d_max = torch.max(cam_d.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1) + 1e-5
            cam_d_norm = F.relu(cam_d - 1e-5) / cam_d_max
            cam_d_norm[:, -1, :, :] = 1 - torch.max(cam_d_norm[:, :-1, :, :], dim=1)[0]
            cam_max = torch.max(cam_d_norm[:, :-1, :, :], dim=1, keepdim=True)[0]
            cam_d_norm[:, :-1, :, :][cam_d_norm[:, :-1, :, :] < cam_max] = 0

        f8_3_ = F.relu(self.f8_3(d['conv4']), inplace=True)
        f8_4_ = F.relu(self.f8_4(d['conv5']), inplace=True)

        f8_3 = f8_3_.detach()
        f8_4 = f8_4_.detach()
        
        # f8_3_ = F.relu(self.proj8_3(f8_3_), inplace=True)
        # f8_4_ = F.relu(self.proj8_4(f8_4_), inplace=True)

        x_s = F.interpolate(img, (h, w), mode='bilinear', align_corners=True)
        f = torch.cat([x_s, f8_3, f8_4], dim=1)
        feat_cat = torch.cat([f8_3_, f8_4_, feat], dim=1)

        cam_rv = self.PCM(cam_d_norm, f)

        pred = F.avg_pool2d(cam, kernel_size=(h, w), padding=0)
        pred = pred.view(pred.size(0), -1)

        pred_rv = F.avg_pool2d(cam_rv, kernel_size=(h, w), padding=0)
        pred_rv = pred_rv.view(pred_rv.size(0), -1)
        if return_feat:
            return pred, cam, pred_rv, cam_rv, feat_cat
        else:
            return pred, cam, pred_rv, cam_rv

    def forward_cam(self, img):
        d = super().forward_as_dict(img)
        # cam = self.fc8(self.dropout7(d['conv6']))
        # dropout 뺸 것과 동일
        cam = self.fc8(d['conv6'])
        return cam 
    
    def forward_cam_rv(self, img):
        d = super().forward_as_dict(img)
        # cam = self.fc8(self.dropout7(d['conv6']))
        cam = self.fc8(d['conv6'])
        n, c, h, w = cam.size()
        with torch.no_grad():
            cam_d = F.relu(cam.detach())
            cam_d_max = torch.max(cam_d.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1)+1e-5
            cam_d_norm = F.relu(cam_d - 1e-5) / cam_d_max
            
            cam_d_norm[:, -1, :, :] = 1 - torch.max(cam_d_norm[:, :-1, :, :], dim=1)[0]
            cam_max = torch.max(cam_d_norm[:, :-1, :, :], dim=1, keepdim=True)[0]
            cam_d_norm[:, :-1, :, :][cam_d_norm[:, :-1, :, :] < cam_max] = 0
            
            f8_3 = F.relu(self.f8_3(d['conv4'].detach()), inplace=True)
            f8_4 = F.relu(self.f8_4(d['conv5'].detach()), inplace=True)
            x_s = F.interpolate(img, (h, w), mode='bilinear', align_corners=True)
            f = torch.cat([x_s, f8_3, f8_4], dim=1)
            cam_rv = self.PCM(cam_d_norm, f)
        return cam_rv

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups

    def PCM(self, cam, f):
        n, c, h, w = f.size()
        cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h*w)
        f = self.f9(f)
        f = f.view(n, -1, h * w)
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)

        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)
        aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-5)
        cam_rv = torch.matmul(cam, aff).view(n, -1, h, w)

        return cam_rv
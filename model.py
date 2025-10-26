# model.py
import torch
import torch.nn as nn
import torchvision.models as models

class TemporalMobileNet(nn.Module):
    def __init__(self, num_classes, pretrained=True, embed_dim=512):
        super().__init__()
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        # remove classifier, keep features (features -> last_conv -> avgpool -> 1280)
        self.backbone = mobilenet.features  # returns feature map
        # global pool to get vector
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        backbone_out_dim = 1280  # MobileNetV2 default
        self.fc_embed = nn.Linear(backbone_out_dim, embed_dim)
        self.relu = nn.ReLU()
        # classifier on temporal pooled embedding
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        """
        x: tensor (B, T, C, H, W)
        returns logits (B, num_classes)
        """
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)               # (B*T, C, H, W)
        feats = self.backbone(x)               # (B*T, feat, h, w)
        feats = self.global_pool(feats)        # (B*T, feat, 1, 1)
        feats = feats.view(B, T, -1)           # (B, T, feat)
        feats = self.fc_embed(feats)           # (B, T, embed_dim)
        feats = self.relu(feats)
        # temporal pooling (mean)
        pooled = feats.mean(dim=1)             # (B, embed_dim)
        logits = self.classifier(pooled)       # (B, num_classes)
        return logits

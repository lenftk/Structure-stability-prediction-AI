import torch
import torch.nn as nn
import timm

class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super().__init__()
        # Cross attention where Front attends to Top, and Top attends to Front
        self.front_to_top = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.top_to_front = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
    def forward(self, f_front, f_top):
        # f_front, f_top shape: (B, C) -> transform to (B, 1, C) for attention
        f_front_seq = f_front.unsqueeze(1)
        f_top_seq = f_top.unsqueeze(1)
        
        # Front query, Top key/value
        out_front, _ = self.front_to_top(f_front_seq, f_top_seq, f_top_seq)
        # Top query, Front key/value
        out_top, _ = self.top_to_front(f_top_seq, f_front_seq, f_front_seq)
        
        # Add & Norm
        f_front_fused = self.norm1(f_front_seq + out_front).squeeze(1)
        f_top_fused = self.norm2(f_top_seq + out_top).squeeze(1)
        
        return torch.cat((f_front_fused, f_top_fused), dim=1)


class AdvancedStructureModel(nn.Module):
    def __init__(self, backbone_name='convnext_tiny', pretrained=True, num_classes=2):
        super().__init__()
        
        # Use timm for powerful modern backbones
        # e.g., 'convnext_tiny', 'tf_efficientnetv2_s', 'swinv2_tiny_window16_256'
        self.backbone_front = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0) # num_classes=0 for features
        self.backbone_top = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        
        feature_dim = self.backbone_front.num_features
        
        # Cross-Attention Fusion
        self.fusion = CrossAttentionFusion(feature_dim=feature_dim)
        
        # Classifier Head
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, front_img, top_img):
        f_front = self.backbone_front(front_img)
        f_top = self.backbone_top(top_img)
        
        fused_features = self.fusion(f_front, f_top)
        
        logits = self.fc(fused_features)
        return logits

# simple test block
if __name__ == '__main__':
    model = AdvancedStructureModel('resnet34') # Testing with standard resnet
    f = torch.randn(2, 3, 224, 224)
    t = torch.randn(2, 3, 224, 224)
    out = model(f, t)
    print("Output shape:", out.shape)

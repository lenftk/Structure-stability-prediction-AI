import torch
import torch.nn as nn
import timm

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x_q, x_kv):
        attn_out, _ = self.mha(x_q, x_kv, x_kv)
        return self.norm(x_q + attn_out)

class AdvancedCrossAttentionDualNet(nn.Module):
    def __init__(self, model_name='convnext_base.fb_in22k_ft_in1k_384', pretrained=True):
        super(AdvancedCrossAttentionDualNet, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        
        self.cross_attn_f2t = CrossAttention(embed_dim=in_features)
        self.cross_attn_t2f = CrossAttention(embed_dim=in_features)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features * 2, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1)
        )

    def forward(self, front_img, top_img):
        front_feat = self.backbone(front_img)
        top_feat = self.backbone(top_img)
        
        front_feat_seq = front_feat.unsqueeze(1)
        top_feat_seq = top_feat.unsqueeze(1)
        
        f2t = self.cross_attn_f2t(front_feat_seq, top_feat_seq).squeeze(1)
        t2f = self.cross_attn_t2f(top_feat_seq, front_feat_seq).squeeze(1)
        
        fused_feat = torch.cat([f2t, t2f], dim=1)
        
        logits = self.classifier(fused_feat)
        return logits.squeeze(-1)

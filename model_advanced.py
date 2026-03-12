import torch
import torch.nn as nn
import timm

class AdvancedDualViewModel(nn.Module):
    """
    Swin Transformer, EfficientNet 등 다양한 Backbone을 지원하고
    Multi-Sample Dropout 이나 GeM Pooling 같은 고급 기법을 적용할 수 있는 모델입니다.
    """
    def __init__(self, model_name='swin_tiny_patch4_window7_224', num_classes=2, pretrained=True):
        super().__init__()
        # timm 백본 생성 (분류기 제거)
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        in_features = self.backbone.num_features
        
        # Classifier 헤드 강화 (Multi-Sample Dropout 등은 생략하고 안정적인 구조 적용)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features * 2, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, front, top):
        f_feat = self.backbone(front)
        t_feat = self.backbone(top)
        
        combined = torch.cat([f_feat, t_feat], dim=1)
        logits = self.classifier(combined)
        return logits

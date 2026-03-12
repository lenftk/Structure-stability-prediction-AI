import torch
import torch.nn as nn
import torchvision.models as models

class StructureStabilityModel(nn.Module):
    def __init__(self, backbone_name='resnet34', pretrained=True, num_classes=2):
        super(StructureStabilityModel, self).__init__()
        
        # Load backbone
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        elif backbone_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Backbone {backbone_name} not supported yet.")
            
        # Get in_features for the fc layer
        in_features = self.backbone.fc.in_features
        
        # Remove the final fc layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Our fusion network
        # Since we have two views, we'll extract features from both
        # and concatenate them: in_features * 2
        self.fc = nn.Sequential(
            nn.Linear(in_features * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, front_img, top_img):
        # Extract features from front view
        f_front = self.backbone(front_img)
        f_front = torch.flatten(f_front, 1)
        
        # Extract features from top view
        f_top = self.backbone(top_img)
        f_top = torch.flatten(f_top, 1)
        
        # Concatenate features
        features = torch.cat((f_front, f_top), dim=1)
        
        # Predict logits
        logits = self.fc(features)
        return logits

# simple test block
if __name__ == '__main__':
    model = StructureStabilityModel(backbone_name='resnet18')
    f = torch.randn(2, 3, 224, 224)
    t = torch.randn(2, 3, 224, 224)
    out = model(f, t)
    print("Output shape:", out.shape) # Expected (2, 2)

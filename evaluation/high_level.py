"""
High-level metrics: AlexNet, CLIP, InceptionV3
"""

import torch
import torch.nn as nn
import clip
from torchvision.models import alexnet, inception_v3
import numpy as np

class AlexNetMetrics:
    """AlexNet two-way identification accuracy"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.model = alexnet(pretrained=True).to(device).eval()
    
    def extract_features(self, images, layer=2):
        """Extract AlexNet features from specified layer"""
        with torch.no_grad():
            x = images
            for idx, module in enumerate(self.model.features):
                x = module(x)
                # AlexNet features structure:
                # 0: Conv2d(3, 64)
                # 1: ReLU
                # 2: MaxPool2d -> End of Layer 1
                # 3: Conv2d(64, 192)
                # 4: ReLU
                # 5: MaxPool2d -> End of Layer 2 ✅
                # 6: Conv2d(192, 384)
                # 7: ReLU
                # 8: Conv2d(384, 256)
                # 9: ReLU
                # 10: Conv2d(256, 256)
                # 11: ReLU
                # 12: MaxPool2d -> End of Layer 5 ✅
                
                if layer == 2 and idx == 5:  # ✅ Correct for AlexNet-2
                    return x.flatten(1)
                elif layer == 5 and idx == 12:  # ✅ Correct for AlexNet-5
                    return x.flatten(1)
            
            return x.flatten(1)
    
    def two_way_identification(self, recon_features, gt_features):
        """
        Compute two-way identification accuracy (paper-compliant).
        
        For each reconstruction, check if its best matching ground truth
        is the correct one (diagonal element is maximum in its row).
        """
        n_samples = recon_features.shape[0]
        
        # Check for empty or invalid features
        if n_samples == 0 or recon_features.shape[1] == 0 or gt_features.shape[1] == 0:
            print(f"Warning: Invalid features - n_samples: {n_samples}, "
                  f"recon_shape: {recon_features.shape}, gt_shape: {gt_features.shape}")
            return 0.0
        
        # Normalize features
        recon_feat_norm = torch.nn.functional.normalize(recon_features, dim=1)
        gt_feat_norm = torch.nn.functional.normalize(gt_features, dim=1)
        
        # Compute similarity matrix
        similarities = torch.mm(recon_feat_norm, gt_feat_norm.T)
        
        # Two-way identification (paper's method):
        # Check if diagonal elements are maximum in their row
        correct = 0
        for i in range(n_samples):
            if torch.argmax(similarities[i]) == i:
                correct += 1
        
        return correct / n_samples * 100
    
    def compute_layer2(self, recon_batch, gt_batch):
        """Compute AlexNet layer 2 two-way identification"""
        recon_feat = self.extract_features(recon_batch, layer=2)
        gt_feat = self.extract_features(gt_batch, layer=2)
        return self.two_way_identification(recon_feat, gt_feat)
    
    def compute_layer5(self, recon_batch, gt_batch):
        """Compute AlexNet layer 5 two-way identification"""
        recon_feat = self.extract_features(recon_batch, layer=5)
        gt_feat = self.extract_features(gt_batch, layer=5)
        return self.two_way_identification(recon_feat, gt_feat)


class CLIPMetrics:
    """CLIP two-way identification accuracy"""
    
    def __init__(self, device='cuda', model_name='ViT-L/14'):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
    
    def extract_features(self, images):
        """Extract CLIP image features"""
        with torch.no_grad():
            # Convert from [-1, 1] to [0, 1] if needed
            if images.min() < 0:
                images_01 = torch.clamp((images + 1) / 2, 0, 1)
            else:
                images_01 = images
            
            # Resize to 224x224
            images_resized = torch.nn.functional.interpolate(
                images_01, size=(224, 224), mode='bilinear', align_corners=False
            )
            
            # Apply CLIP normalization (CRITICAL!)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)
            images_normalized = (images_resized - mean) / std
            
            features = self.model.encode_image(images_normalized)
            
            return features
    
    def two_way_identification(self, recon_features, gt_features):
        """Compute two-way identification accuracy (paper-compliant)"""
        n_samples = recon_features.shape[0]
        
        # Check for empty or invalid features
        if n_samples == 0 or recon_features.shape[1] == 0 or gt_features.shape[1] == 0:
            return 0.0
        
        # Normalize features
        recon_feat_norm = torch.nn.functional.normalize(recon_features, dim=1)
        gt_feat_norm = torch.nn.functional.normalize(gt_features, dim=1)
        
        # Compute similarity matrix
        similarities = torch.mm(recon_feat_norm, gt_feat_norm.T)
        
        # Two-way identification (paper's method)
        correct = 0
        for i in range(n_samples):
            if torch.argmax(similarities[i]) == i:
                correct += 1
        
        return correct / n_samples * 100
    
    def compute(self, recon_batch, gt_batch):
        """Compute CLIP two-way identification"""
        recon_feat = self.extract_features(recon_batch)
        gt_feat = self.extract_features(gt_batch)
        return self.two_way_identification(recon_feat, gt_feat)


class InceptionMetrics:
    """InceptionV3 two-way identification accuracy"""
    
    def __init__(self, device='cuda'):
        self.device = device
        # Load InceptionV3 (aux_logits defaults to True)
        self.model = inception_v3(pretrained=True).to(device).eval()
        # Remove final fc layer to get features
        self.model.fc = torch.nn.Identity()
    
    def extract_features(self, images):
        """Extract InceptionV3 features"""
        with torch.no_grad():
            # Resize to 299x299
            images_resized = torch.nn.functional.interpolate(
                images, size=(299, 299), mode='bilinear', align_corners=False
            )
            
            # Apply ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
            images_normalized = (images_resized - mean) / std
            
            # Extract features
            features = self.model(images_normalized)
            
            return features
    
    def two_way_identification(self, recon_features, gt_features):
        """Compute two-way identification accuracy (paper-compliant)"""
        n_samples = recon_features.shape[0]
        
        # Check for empty or invalid features
        if n_samples == 0 or recon_features.shape[1] == 0 or gt_features.shape[1] == 0:
            return 0.0
        
        # Normalize features
        recon_feat_norm = torch.nn.functional.normalize(recon_features, dim=1)
        gt_feat_norm = torch.nn.functional.normalize(gt_features, dim=1)
        
        # Compute similarity matrix
        similarities = torch.mm(recon_feat_norm, gt_feat_norm.T)
        
        # Two-way identification (paper's method)
        correct = 0
        for i in range(n_samples):
            if torch.argmax(similarities[i]) == i:
                correct += 1
        
        return correct / n_samples * 100
    
    def compute(self, recon_batch, gt_batch):
        """Compute two-way identification accuracy"""
        recon_feat = self.extract_features(recon_batch)
        gt_feat = self.extract_features(gt_batch)
        return self.two_way_identification(recon_feat, gt_feat)

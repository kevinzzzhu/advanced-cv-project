"""
Image quality metrics: Inception Score (IS) and Fréchet Inception Distance (FID)
"""
import torch
import numpy as np
from torchvision.models import inception_v3
from scipy import linalg


class InceptionScore:
    """Inception Score (IS) for image quality assessment"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.model = inception_v3(pretrained=True, transform_input=False).to(device).eval()
    
    def get_inception_predictions(self, images):
        """Get InceptionV3 predictions for images"""
        with torch.no_grad():
            # Ensure images are in correct format
            if images.dim() == 3:
                images = images.unsqueeze(0)
            
            # Resize to 299x299 for InceptionV3
            images = torch.nn.functional.interpolate(
                images, size=(299, 299), mode='bilinear', align_corners=False
            )
            
            # Get predictions
            predictions = self.model(images)
            # Apply softmax to get probabilities
            predictions = torch.nn.functional.softmax(predictions, dim=1)
            
        return predictions.cpu().numpy()
    
    def compute_inception_score(self, images):
        """Compute Inception Score"""
        predictions = self.get_inception_predictions(images)
        
        # Compute marginal distribution
        marginal = np.mean(predictions, axis=0)
        
        # Compute KL divergence for each image
        kl_divs = []
        for pred in predictions:
            kl_div = np.sum(pred * np.log(pred / marginal + 1e-10))
            kl_divs.append(kl_div)
        
        # Inception Score is exp of mean KL divergence
        is_score = np.exp(np.mean(kl_divs))
        return is_score
    
    def compute(self, images):
        """Compute Inception Score for a batch of images"""
        return self.compute_inception_score(images)


class FID:
    """Fréchet Inception Distance (FID) for image quality assessment"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.model = inception_v3(pretrained=True, transform_input=False).to(device).eval()
    
    def get_inception_features(self, images):
        """Extract InceptionV3 features for FID computation"""
        with torch.no_grad():
            # Ensure images are in correct format
            if images.dim() == 3:
                images = images.unsqueeze(0)
            
            # Resize to 299x299 for InceptionV3
            images = torch.nn.functional.interpolate(
                images, size=(299, 299), mode='bilinear', align_corners=False
            )
            
            # Get features from the last pooling layer
            features = self.model.Conv2d_1a_3x3(images)
            features = self.model.Conv2d_2a_3x3(features)
            features = self.model.Conv2d_2b_3x3(features)
            features = self.model.maxpool1(features)
            features = self.model.Conv2d_3b_1x1(features)
            features = self.model.Conv2d_4a_3x3(features)
            features = self.model.maxpool2(features)
            features = self.model.Mixed_5b(features)
            features = self.model.Mixed_5c(features)
            features = self.model.Mixed_5d(features)
            features = self.model.Mixed_6a(features)
            features = self.model.Mixed_6b(features)
            features = self.model.Mixed_6c(features)
            features = self.model.Mixed_6d(features)
            features = self.model.Mixed_6e(features)
            features = self.model.Mixed_7a(features)
            features = self.model.Mixed_7b(features)
            features = self.model.Mixed_7c(features)
            features = self.model.avgpool(features)
            features = features.view(features.size(0), -1)
            
        return features.cpu().numpy()
    
    def compute_fid(self, real_features, generated_features):
        """Compute FID between real and generated features"""
        # Compute mean and covariance
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
        
        # Compute FID
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % 1e-6
            print(msg)
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        
        return fid
    
    def compute(self, real_images, generated_images):
        """Compute FID between real and generated images"""
        real_features = self.get_inception_features(real_images)
        generated_features = self.get_inception_features(generated_images)
        return self.compute_fid(real_features, generated_features)

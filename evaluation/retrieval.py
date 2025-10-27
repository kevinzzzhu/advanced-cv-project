"""
Retrieval metrics: Image and Brain retrieval
"""
import torch
import numpy as np
from scipy.spatial.distance import cdist


class ImageRetrieval:
    """Image retrieval: Given fMRI, select most similar image from candidates"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.candidate_images = None
        self.candidate_features = None
    
    def load_candidates(self, candidate_images, feature_extractor):
        """Load candidate images and extract their features"""
        self.candidate_images = candidate_images
        print(f"Extracting features for {len(candidate_images)} candidate images...")
        
        # Extract features for all candidate images
        features = []
        for img in candidate_images:
            if isinstance(img, str):
                # Load image from path
                from PIL import Image
                img = Image.open(img)
            
            # Convert to tensor and extract features
            img_tensor = self._preprocess_image(img)
            with torch.no_grad():
                feat = feature_extractor(img_tensor.unsqueeze(0).to(self.device))
                features.append(feat.cpu().numpy())
        
        self.candidate_features = np.vstack(features)
        print(f"Loaded {len(self.candidate_features)} candidate features")
    
    def _preprocess_image(self, image):
        """Preprocess image for feature extraction"""
        if hasattr(image, 'convert'):
            image = image.convert('RGB')
        
        # Convert to tensor and normalize
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if hasattr(image, 'convert'):
            return transform(image)
        else:
            return image  # Already a tensor
    
    def retrieve(self, query_fmri, fmri_to_image_mapper, top_k=1):
        """Retrieve most similar images for given fMRI"""
        if self.candidate_features is None:
            raise ValueError("Candidate features not loaded. Call load_candidates() first.")
        
        # Map fMRI to image features
        query_features = fmri_to_image_mapper(query_fmri)
        
        # Compute similarities
        similarities = np.dot(query_features, self.candidate_features.T)
        
        # Get top-k most similar
        top_indices = np.argsort(similarities, axis=1)[:, -top_k:]
        
        return top_indices, similarities
    
    def compute_accuracy(self, test_fmri, ground_truth_indices, fmri_to_image_mapper):
        """Compute retrieval accuracy"""
        top_indices, _ = self.retrieve(test_fmri, fmri_to_image_mapper, top_k=1)
        
        correct = 0
        for i, gt_idx in enumerate(ground_truth_indices):
            if gt_idx in top_indices[i]:
                correct += 1
        
        return correct / len(ground_truth_indices) * 100


class BrainRetrieval:
    """Brain retrieval: Given image, select most similar fMRI from candidates"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.candidate_fmri = None
        self.candidate_features = None
    
    def load_candidates(self, candidate_fmri, fmri_feature_extractor):
        """Load candidate fMRI scans and extract their features"""
        self.candidate_fmri = candidate_fmri
        print(f"Extracting features for {len(candidate_fmri)} candidate fMRI scans...")
        
        # Extract features for all candidate fMRI
        features = []
        for fmri in candidate_fmri:
            with torch.no_grad():
                feat = fmri_feature_extractor(fmri)
                features.append(feat.cpu().numpy())
        
        self.candidate_features = np.vstack(features)
        print(f"Loaded {len(self.candidate_features)} candidate fMRI features")
    
    def retrieve(self, query_image, image_to_fmri_mapper, top_k=1):
        """Retrieve most similar fMRI for given image"""
        if self.candidate_features is None:
            raise ValueError("Candidate fMRI features not loaded. Call load_candidates() first.")
        
        # Map image to fMRI features
        query_features = image_to_fmri_mapper(query_image)
        
        # Compute similarities
        similarities = np.dot(query_features, self.candidate_features.T)
        
        # Get top-k most similar
        top_indices = np.argsort(similarities, axis=1)[:, -top_k:]
        
        return top_indices, similarities
    
    def compute_accuracy(self, test_images, ground_truth_indices, image_to_fmri_mapper):
        """Compute retrieval accuracy"""
        top_indices, _ = self.retrieve(test_images, image_to_fmri_mapper, top_k=1)
        
        correct = 0
        for i, gt_idx in enumerate(ground_truth_indices):
            if gt_idx in top_indices[i]:
                correct += 1
        
        return correct / len(ground_truth_indices) * 100

import torch
import os
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision.models import ResNet50_Weights
import io

# Full PlantVillage dataset with 38 classes
DISEASE_CLASSES = {
    0: {
        "name": "Apple___Apple_scab",
        "scientific_name": "Venturia inaequalis",
        "symptoms": "Dark brown to black lesions with rough, scabby texture on leaves and fruit.",
        "treatment": "Apply fungicides, remove fallen leaves, and use resistant varieties."
    },
    1: {
        "name": "Apple___Black_rot",
        "scientific_name": "Botryosphaeria obtusa",
        "symptoms": "Circular lesions with purple margins on leaves, cankers on branches.",
        "treatment": "Prune infected branches, apply fungicides, and maintain tree health."
    },
    2: {
        "name": "Apple___Cedar_apple_rust",
        "scientific_name": "Gymnosporangium juniperi-virginianae",
        "symptoms": "Bright orange-yellow spots on leaves with red borders.",
        "treatment": "Remove nearby cedar trees, apply fungicides, and use resistant varieties."
    },
    3: {
        "name": "Apple___healthy",
        "scientific_name": "Malus domestica (healthy)",
        "symptoms": "Leaves are green and healthy with no visible disease symptoms.",
        "treatment": "Maintain proper care and monitoring."
    },
    4: {
        "name": "Blueberry___healthy",
        "scientific_name": "Vaccinium corymbosum (healthy)",
        "symptoms": "Healthy blueberry plant with no disease symptoms.",
        "treatment": "Continue proper maintenance and care."
    },
    5: {
        "name": "Cherry_(including_sour)___Powdery_mildew",
        "scientific_name": "Podosphaera clandestina",
        "symptoms": "White powdery growth on leaves, shoots, and flowers.",
        "treatment": "Apply fungicides, improve air circulation, and use resistant varieties."
    },
    6: {
        "name": "Cherry_(including_sour)___healthy",
        "scientific_name": "Prunus avium/cerasus (healthy)",
        "symptoms": "Healthy cherry tree with no disease symptoms.",
        "treatment": "Maintain proper care and monitoring."
    },
    7: {
        "name": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "scientific_name": "Cercospora zeae-maydis",
        "symptoms": "Gray to tan lesions with dark borders on leaves.",
        "treatment": "Crop rotation, tillage, and fungicide application."
    },
    8: {
        "name": "Corn_(maize)___Common_rust_",
        "scientific_name": "Puccinia sorghi",
        "symptoms": "Reddish-brown pustules on both sides of leaves.",
        "treatment": "Use resistant hybrids and apply fungicides when necessary."
    },
    9: {
        "name": "Corn_(maize)___Northern_Leaf_Blight",
        "scientific_name": "Exserohilum turcicum",
        "symptoms": "Long, elliptical lesions with tan centers and dark borders.",
        "treatment": "Crop rotation, tillage, and fungicide application."
    },
    10: {
        "name": "Corn_(maize)___healthy",
        "scientific_name": "Zea mays (healthy)",
        "symptoms": "Healthy corn plant with no disease symptoms.",
        "treatment": "Continue proper maintenance and care."
    },
    11: {
        "name": "Grape___Black_rot",
        "scientific_name": "Guignardia bidwellii",
        "symptoms": "Circular lesions with dark borders and black fruiting bodies.",
        "treatment": "Remove infected plant parts, apply fungicides, and improve air circulation."
    },
    12: {
        "name": "Grape___Esca_(Black_Measles)",
        "scientific_name": "Phaeomoniella chlamydospora",
        "symptoms": "Dark spots on leaves and berries, wood decay in older vines.",
        "treatment": "Remove infected vines and maintain vineyard health."
    },
    13: {
        "name": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "scientific_name": "Pseudocercospora vitis",
        "symptoms": "Angular brown lesions with dark borders on leaves.",
        "treatment": "Apply fungicides and improve air circulation."
    },
    14: {
        "name": "Grape___healthy",
        "scientific_name": "Vitis vinifera (healthy)",
        "symptoms": "Healthy grapevine with no disease symptoms.",
        "treatment": "Maintain proper care and monitoring."
    },
    15: {
        "name": "Orange___Haunglongbing_(Citrus_greening)",
        "scientific_name": "Candidatus Liberibacter asiaticus",
        "symptoms": "Yellowing of leaves, stunted growth, and misshapen fruit.",
        "treatment": "Remove infected trees and control psyllid vectors."
    },
    16: {
        "name": "Peach___Bacterial_spot",
        "scientific_name": "Xanthomonas arboricola pv. pruni",
        "symptoms": "Small, dark lesions on leaves, stems, and fruit.",
        "treatment": "Use resistant varieties and apply copper-based bactericides."
    },
    17: {
        "name": "Peach___healthy",
        "scientific_name": "Prunus persica (healthy)",
        "symptoms": "Healthy peach tree with no disease symptoms.",
        "treatment": "Maintain proper care and monitoring."
    },
    18: {
        "name": "Pepper,_bell___Bacterial_spot",
        "scientific_name": "Xanthomonas campestris pv. vesicatoria",
        "symptoms": "Small, water-soaked spots that enlarge and turn brown with yellow halos.",
        "treatment": "Use disease-free seeds, practice crop rotation, and apply copper-based fungicides."
    },
    19: {
        "name": "Pepper,_bell___healthy",
        "scientific_name": "Capsicum annuum (healthy)",
        "symptoms": "Healthy pepper plant with no disease symptoms.",
        "treatment": "Maintain proper care and monitoring."
    },
    20: {
        "name": "Potato___Early_blight",
        "scientific_name": "Alternaria solani",
        "symptoms": "Dark brown to black lesions with concentric rings on leaves.",
        "treatment": "Crop rotation, proper plant spacing, and timely application of fungicides."
    },
    21: {
        "name": "Potato___Late_blight",
        "scientific_name": "Phytophthora infestans",
        "symptoms": "Irregular, water-soaked lesions that rapidly enlarge and turn brown/black.",
        "treatment": "Remove infected plants, use resistant varieties, and apply preventative fungicides."
    },
    22: {
        "name": "Potato___healthy",
        "scientific_name": "Solanum tuberosum (healthy)",
        "symptoms": "Healthy potato plant with no disease symptoms.",
        "treatment": "Continue proper maintenance and care."
    },
    23: {
        "name": "Raspberry___healthy",
        "scientific_name": "Rubus idaeus (healthy)",
        "symptoms": "Healthy raspberry plant with no disease symptoms.",
        "treatment": "Maintain proper care and monitoring."
    },
    24: {
        "name": "Soybean___healthy",
        "scientific_name": "Glycine max (healthy)",
        "symptoms": "Healthy soybean plant with no disease symptoms.",
        "treatment": "Continue proper maintenance and care."
    },
    25: {
        "name": "Squash___Powdery_mildew",
        "scientific_name": "Podosphaera xanthii",
        "symptoms": "White powdery growth on leaves and stems.",
        "treatment": "Apply fungicides, improve air circulation, and use resistant varieties."
    },
    26: {
        "name": "Strawberry___Leaf_scorch",
        "scientific_name": "Diplocarpon earliana",
        "symptoms": "Dark purple spots on leaves that can cause defoliation.",
        "treatment": "Remove infected leaves, apply fungicides, and improve air circulation."
    },
    27: {
        "name": "Strawberry___healthy",
        "scientific_name": "Fragaria × ananassa (healthy)",
        "symptoms": "Healthy strawberry plant with no disease symptoms.",
        "treatment": "Maintain proper care and monitoring."
    },
    28: {
        "name": "Tomato___Bacterial_spot",
        "scientific_name": "Xanthomonas campestris pv. vesicatoria",
        "symptoms": "Small, dark, greasy spots on leaves and raised lesions on fruit.",
        "treatment": "Use disease-free seeds, rotate crops, and apply copper-based bactericides."
    },
    29: {
        "name": "Tomato___Early_blight",
        "scientific_name": "Alternaria solani",
        "symptoms": "Dark brown spots with concentric rings on older leaves.",
        "treatment": "Remove infected lower leaves, improve air circulation, and apply fungicides."
    },
    30: {
        "name": "Tomato___Late_blight",
        "scientific_name": "Phytophthora infestans",
        "symptoms": "Large, water-soaked lesions that rapidly expand and turn brown.",
        "treatment": "Remove infected plants, use resistant varieties, and apply fungicides preventatively."
    },
    31: {
        "name": "Tomato___Leaf_Mold",
        "scientific_name": "Fulvia fulva",
        "symptoms": "Pale green or yellow spots on upper leaf surfaces with velvety patches underneath.",
        "treatment": "Improve air circulation, reduce humidity, and use resistant varieties."
    },
    32: {
        "name": "Tomato___Septoria_leaf_spot",
        "scientific_name": "Septoria lycopersici",
        "symptoms": "Small, circular, dark spots with tiny black dots in the center.",
        "treatment": "Remove infected leaves, avoid overhead watering, and apply fungicides."
    },
    33: {
        "name": "Tomato___Spider_mites Two-spotted_spider_mite",
        "scientific_name": "Tetranychus urticae",
        "symptoms": "Tiny yellow or white spots on leaves with fine webbing underneath.",
        "treatment": "Hose down plants, use insecticidal soaps, and introduce predatory mites."
    },
    34: {
        "name": "Tomato___Target_Spot",
        "scientific_name": "Corynespora cassiicola",
        "symptoms": "Small, dark, circular spots with light brown centers and dark borders.",
        "treatment": "Crop rotation, proper sanitation, and fungicide application."
    },
    35: {
        "name": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "scientific_name": "Tomato yellow leaf curl virus (TYLCV)",
        "symptoms": "Leaves become small, curled upwards, and turn yellow with purple veins.",
        "treatment": "Manage whitefly vectors, use resistant varieties, and remove infected plants."
    },
    36: {
        "name": "Tomato___Tomato_mosaic_virus",
        "scientific_name": "Tomato mosaic virus (ToMV)",
        "symptoms": "Mosaic patterns of light and dark green on leaves with distortion.",
        "treatment": "Use resistant varieties, practice good sanitation, and remove infected plants."
    },
    37: {
        "name": "Tomato___healthy",
        "scientific_name": "Solanum lycopersicum (healthy)",
        "symptoms": "Healthy tomato plant with no disease symptoms.",
        "treatment": "Maintain proper care and monitoring."
    }
}

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes=None):
        super(PlantDiseaseModel, self).__init__()
        # Load pre-trained ResNet50 with improved weights
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Unfreeze more layers for better fine-tuning
        for name, param in self.model.named_parameters():
            if "layer3" not in name and "layer4" not in name and "fc" not in name:
                param.requires_grad = False
        
        # Enhanced model architecture for plant disease classification
        num_features = self.model.fc.in_features
        
        # Use provided num_classes or default to DISEASE_CLASSES length
        if num_classes is None:
            num_classes = len(DISEASE_CLASSES)
        
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, num_classes)
        )
        
        # Initialize weights properly
        for m in self.model.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass of the model"""
        return self.model(x)

    def train_model(self, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
        """Train the model on plant disease dataset"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100. * correct / total

            # Validation phase
            self.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for v_batch_idx, (images, labels) in enumerate(val_loader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = self(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Save inside backend/app/models/best_model.pth
                models_dir = os.path.dirname(__file__)
                save_path = os.path.join(models_dir, 'best_model.pth')
                torch.save(self.state_dict(), save_path)

    def predict(self, image_tensor):
        """Make prediction with improved confidence calculation"""
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            outputs = self(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get top 5 predictions
            top_probs, top_indices = torch.topk(probabilities[0], k=min(5, outputs.size(1)))
            
            # Convert to percentages and create prediction list
            predictions = []
            total_prob = torch.sum(probabilities[0])
            
            for prob, idx in zip(top_probs, top_indices):
                normalized_prob = (prob / total_prob) * 100
                predictions.append((idx.item(), float(normalized_prob)))
            
            print(f"Debug - Top 5 Predictions: {predictions}")
            return predictions

    def preprocess_image(self, image_bytes):
        """Preprocess the image with improved augmentation."""
        try:
            # Open image and convert to RGB
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Define preprocessing pipeline
            preprocess = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # Apply preprocessing
            image_tensor = preprocess(image)
            return image_tensor.unsqueeze(0)
            
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None

def get_transform():
    """Get the transformation pipeline for input images"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def predict_disease(model, image):
    """
    Predict the disease from an image and return data formatted for the frontend.
    """
    try:
        model.eval()
        transform = get_transform()
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get all predictions sorted by probability
            values, indices = torch.sort(probabilities, dim=1, descending=True)
            values = values[0]
            indices = indices[0]
            
            # Calculate more robust confidence metrics
            top_prob = values[0].item()
            second_prob = values[1].item()
            mean_other_probs = torch.mean(values[1:]).item()
            std_other_probs = torch.std(values[1:]).item()
            
            # Multiple confidence checks
            prob_diff = top_prob - second_prob
            prob_ratio = top_prob / (second_prob + 1e-6)
            distribution_score = (top_prob - mean_other_probs) / (std_other_probs + 1e-6)
            
            # Combined confidence calculation with adjusted weights
            confidence_score = (
                (prob_diff * 0.5) +  # Increased weight for probability difference
                (prob_ratio * 0.3) +  # Weight for probability ratio
                (distribution_score * 0.2)  # Reduced weight for distribution score
            ) * 100
            
            # Get prediction details
            disease_idx = indices[0].item()
            
            # Use the DISEASE_CLASSES dictionary for proper disease information
            
            # Get the disease name from the class index
            if disease_idx in DISEASE_CLASSES:
                disease_name = DISEASE_CLASSES[disease_idx]["name"]
                scientific_name = DISEASE_CLASSES[disease_idx]["scientific_name"]
                symptoms = DISEASE_CLASSES[disease_idx]["symptoms"]
                treatment = DISEASE_CLASSES[disease_idx]["treatment"]
            else:
                disease_name = f"Class_{disease_idx}"
                scientific_name = "Unknown"
                symptoms = "Unable to determine symptoms for this class."
                treatment = "Consult with a plant disease specialist."
            
            # Add debug information with more detail
            predictions = [(idx.item(), prob.item() * 100) 
                         for idx, prob in zip(indices[:5], values[:5])]  # Show top 5 predictions
            print(f"Debug - Top 5 Predictions: {predictions}")
            print(f"Debug - Confidence Score: {confidence_score:.2f}%")
            print(f"Debug - Probability Distribution: diff={prob_diff:.3f}, ratio={prob_ratio:.3f}, dist_score={distribution_score:.3f}")
            
            # Create meaningful disease information
            disease_info = {
                "name": disease_name,
                "scientific_name": scientific_name,
                "symptoms": symptoms,
                "treatment": treatment
            }

            return {
                "disease_name": disease_info["name"],
                "confidence": float(min(100, max(0, confidence_score))),
                "scientific_name": disease_info["scientific_name"],
                "symptoms": disease_info["symptoms"],
                "treatment": disease_info["treatment"],
                "all_predictions": [
                    {
                        "disease": DISEASE_CLASSES[idx.item()]["name"] if idx.item() in DISEASE_CLASSES else f"Class_{idx.item()}",
                        "confidence": float(round(prob.item() * 100, 2))
                    } for idx, prob in zip(indices[:5], values[:5])
                ]
            }
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return {
            "disease_name": "Unknown",
            "confidence": 0.0,
            "scientific_name": "N/A",
            "symptoms": "Unable to process the image.",
            "treatment": "Try a clearer image with good lighting.",
            "all_predictions": []
        }
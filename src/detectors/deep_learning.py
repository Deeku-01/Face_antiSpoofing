import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np
import cv2

class DeepLearningDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", use_fast=True)
        
        # Initialize model with correct number of labels
        self.model = AutoModelForImageClassification.from_pretrained(
            "microsoft/resnet-50",
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def preprocess_image(self, frame, face_rect):
        try:
            x, y, w, h = face_rect
            face_img = frame[y:y+h, x:x+w]
            
            # Resize to match model's expected input size
            face_img = cv2.resize(face_img, (224, 224))
            
            # Convert to RGB if needed
            if len(face_img.shape) == 2:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
            elif face_img.shape[2] == 4:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGRA2RGB)
            elif face_img.shape[2] == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Process image using the processor
            inputs = self.processor(face_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            return inputs
            
        except Exception as e:
            print(f"Error in image preprocessing: {str(e)}")
            return None
    
    def analyze_frame(self, frame, face_rect):
        try:
            inputs = self.preprocess_image(frame, face_rect)
            if inputs is None:
                return {
                    'is_real': False,
                    'confidence': 0.0,
                    'error': 'Preprocessing failed'
                }
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                
                # Get the probability of being real (class 1)
                real_prob = probabilities[0][1].item()
                
                return {
                    'is_real': real_prob > 0.5,
                    'confidence': real_prob,
                    'error': None
                }
                
        except Exception as e:
            print(f"Error in deep learning analysis: {str(e)}")
            return {
                'is_real': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def train(self, train_data, train_labels, epochs=10, batch_size=32):
        """Train the classifier on custom data."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(**batch_data)
                loss = criterion(outputs.logits, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_data)}") 
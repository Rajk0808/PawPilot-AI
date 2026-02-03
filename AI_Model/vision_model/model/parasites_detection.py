
#### NEED TO BE FIXED
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

def load_model(image_path):
    """
    Load a pre-trained EfficientNet-B3 model and make predictions on an image.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (prediction_label, confidence_score)
    """
    model_path = r'AI_Model\vision_model\model\models\efficientnet-b3-parasite-98-77.pth'
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    # Check if image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")
    
    # FIX: Create the model architecture FIRST
    # EfficientNet-B3 with 9 output classes (your parasite classes)
    model = models.efficientnet_b3(pretrained=False)
    
    # Modify the classifier for 9 classes
    num_ftrs = model.classifier[1].in_features if hasattr(model.classifier[1], 'in_features') else model.classifier[0].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 9)
    
    # FIX: Load the saved weights into the model
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # If checkpoint has model_state_dict key
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict):
            # If checkpoint is just state_dict (OrderedDict)
            model.load_state_dict(checkpoint)
        else:
            raise ValueError("Checkpoint format not recognized")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")
    
    model.eval()  # Set to evaluation mode
    
    # Transform pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    # Class names
    class_names = [
        'bloated_belly',
        'flea',
        'healthy cat',
        'healthy dog',
        'mite',
        'scooting_behavior',
        'tick',
        'visible_worms',
        'weight_loss'
    ]
    
    # Load and process image
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {e}")
    
    img = transform(img)  # img is now a tensor
    img = img.unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad(): 
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    
    # Return prediction and confidence
    return class_names[int(pred)], probs[0][int(pred)].item()


if __name__ == "__main__":
    image_path = r'AI_Model\vision_model\data\parasite detection\Screenshot_2026-01-21_153332.png'
    
    try:
        prediction, confidence = load_model(image_path)
        print(f'Prediction: {prediction}, Confidence: {confidence:.4f}')
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
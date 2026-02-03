import torch
import torch.nn as nn
import traceback
import clip

from src.database import model  

def load_model_toy():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_path = "D:\\codeSutra AI\\project\\AI_chat\\AI_Model\\vision_model\\model\\models\\toy_clip_model_final.pth" # Or your local path
    # Recreate classifier
    num_classes = 77
    classifier = nn.Linear(512, num_classes).to(device)
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval() 
    checkpoint = torch.load(load_path, map_location=device)
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    label2id = checkpoint["label2id"]
    id2label = checkpoint["id2label"]
    
    classifier.eval()
    model.eval()
    print("✅ Model and Classifier loaded successfully")

    return model, preprocess, classifier, id2label, device


def predict_toy(image_input, preprocess, model, classifier, id2label, device):
    import torch.nn.functional as F
    from AI_Model.vision_model.utils.load_images import LoadImages
    loader = LoadImages()
    images = loader.image_loader('PIL', [image_input])
    image_tensor = [preprocess(image).unsqueeze(0).to(device) for image in images]

    with torch.no_grad():
        features = model.encode_image(torch.cat(image_tensor, dim=0))
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.float()  

        logits = classifier(features)
        probs = F.softmax(logits, dim=1)
        top_probs, top_ids = probs.topk(3, dim=1)

    results = []
    for i in range(3):
        results.append({
            "label": id2label[top_ids[0][i].item()],
            "confidence": top_probs[0][i].item()
        })

    return results
# --- Usage ---
if __name__ == "__main__":
    try:
        model, preprocess, classifier, id2label, device = load_model_toy()
        image_path = r"AI_Model/vision_model/data/wound avulsion.jpg"

        result = predict_toy(image_path, preprocess, model, classifier, id2label, device)
        print(f"Prediction: {result}")
    except Exception:
        traceback.print_exc()
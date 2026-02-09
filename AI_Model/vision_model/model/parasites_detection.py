from gradio_client import Client, handle_file
from PIL import Image
import tempfile

client = Client("Codesutra/parasite-detection")

def predict_parasite(image_path):

    img = Image.open(image_path).convert("RGB")
    img = img.resize((224,224))

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, "JPEG")

    result = client.predict(handle_file(tmp.name))

    return result


def predict(images):
    result_classes = []
    result_confidence = []
    if len(images) > 1:
        for img in images:
            result = predict_parasite(img)
            result = result.split()
            result[1] = result[1].replace("(","")
            result[1] = result[1].replace(")","")
            result_classes.append(result[0])
            result_confidence.append(float(result[1]))
    else:
        result = predict_parasite(images[0])
        result = result.split()
        result[1] = result[1].replace("(","")
        result[1] = result[1].replace(")","")
        result_classes = result[0]
        result_confidence = float(result[1])
    return result_classes, result_confidence
#if __name__ == "__main__":
#    image_path = 'AI_Model\\vision_model\\data\\parasite detection\\Screenshot_2026-01-21_153332.png'
#    
#    try:
#        
#        print(f"Predicted Parasite: {result[0]}, Confidence: {result[1]}")
#    except FileNotFoundError as e:
#        print(f"Error: {e}")
#    except RuntimeError as e:
#        print(f"Error: {e}")
#    except Exception as e:
#        print(f"Unexpected error: {e}")
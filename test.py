# --- Usage ---
from AI_Model.vision_model.model.packaged_product_scanner import process_food_image
import traceback


if __name__ == "__main__":
    try:
        from PIL import Image
        import io
        #model, preprocess, classifier, id2label, device = load_model()
        image_paths = [
            r"AI_Model/vision_model/data/product/cat treat back.jpg",
            r"AI_Model/vision_model/data/product/cat treat front.jpg"
        ]
        result= process_food_image(image_paths)
        print("Structured Output:", result)
    except Exception:
        traceback.print_exc()
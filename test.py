# --- Usage ---
from AI_Model.vision_model.model.full_body_scan import chatbot_full_body_scan
import traceback
if __name__ == "__main__":
    try:

        prompt = f"""
              Task:
              Estimate the dog's weight range (not exact weight) using all provided images.
              
              Instructions:
              - Use only the provided images and the above context
              - Identify the breed or closest visual match (confirm or note deviations)
              - Estimate adult size category (toy / small / medium / large) 
              - Assess body condition (underweight / ideal / overweight)
              - Assign a body condition score (1 to 9)
              - Use visual cues such as body proportions, limb thickness, chest width, waist definition, and fat coverage
              - Do not assume measurements that are not visible
              - Clearly state assumptions and uncertainty

              Output format (strict):
              - Breed (or closest match):
              - Estimated adult size category:
              - Body condition score (1 to 9):
              - Estimated weight range (kg):
              - Confidence level (low / medium / high):
              - Obesity/underweight:
              - Muscle Loss :
              - Limping or joint stifness :
              - Senior posture issue : 
              - Coat quality issues :
              - anxiety/illness posture :
              - Reasoning based on visible features:
              
              """
        #model, preprocess, classifier, id2label, device = load_model()
        image_paths = [
            r"AI_Model/vision_model/data/full body scan/chihuahua/back side chihuahua.JPG",
            r"AI_Model/vision_model/data/full body scan/chihuahua/front side chihuahua.JPG",
            r"AI_Model/vision_model/data/full body scan/chihuahua/left side angle chihuahua.JPG",
            r"AI_Model/vision_model/data/full body scan/chihuahua/right side angle chihuahua.JPG"
        ]
        result= chatbot_full_body_scan(prompt, image_paths)
        print(result)
    except Exception:
        traceback.print_exc()
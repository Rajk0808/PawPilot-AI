import google.genai as genai
from google.genai import types
import base64
import re
from dotenv import load_dotenv
load_dotenv()
def markdown_bold_to_html(text):
    return re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
def base64_encode_image(image_path):
    # image_path is now a file-like object (InMemoryUploadedFile)
    image_path.seek(0)
    return base64.b64encode(image_path.read()).decode('utf-8')
def create_content(prompt, images):
    parts = [types.Part(text=prompt)]
    for img in images:
        # img is an InMemoryUploadedFile
        img.seek(0)
        parts.append(
            types.Part(
                inline_data=types.Blob(
                    mime_type="image/jpeg",
                    data=base64.b64decode(base64_encode_image(img))
                )
            )
        )
    return types.Content(parts=parts)
def chatbot_emotion_detection(user_query, images):
    # images is a list of InMemoryUploadedFile objects
    client = genai.Client(http_options={'api_version': 'v1alpha'}, api_key=os.getenv("GEMINI_API"))
    prompt = f'''You are a Veternian AI and you have to detect the emotion of the pet in the image and give suggestions to the owner accordingly. if multiple images are provided and prediceted emotion for each image is same then provide a single response otherwise provide response for each image separately.
                by noticing the facial expressions and body language of the dog in the image.
                after that give a brief explanation of 
                Your response should be structured in the following way:
                
                1. Emotional meaning
                2. Training Tips
                3. Comforting actions
                4. Environmental adjustments
                
                User Query: {user_query}'''
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=create_content(prompt, images)
    )
    # Extract the text from the response object
    reply_text = response.text if hasattr(response, 'text') else str(response)
    reply_text = markdown_bold_to_html(reply_text)
    return reply_text

if __name__ == "__main__":
    # Example usage
    with open("AI_Model/vision_model/data/emotion detection/anger.jpeg", "rb") as img_file:
        images = [img_file]
        user_query = "Detect the emotion of the pet in the image and give suggestions to the owner accordingly."
        reply = chatbot_emotion_detection(user_query, images)
        print(reply)

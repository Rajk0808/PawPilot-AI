from langchain_google_genai import ChatGoogleGenerativeAI
from google.genai import types
from langchain_openai import OpenAI
import os
import base64
from dotenv import load_dotenv

load_dotenv()


def base64_encode_image(image_path):
    image_path.seek(0)
    return base64.b64encode(image_path.read()).decode('utf-8')

def create_content(prompt, image):
    parts = [types.Part(text=prompt)]
    for img in image:
        img.seek(0)
        parts.append(
            types.Part(
                inline_data=types.Blob(
                    mime_type='image/jpeg',
                    data=base64.b64decode(base64_encode_image(img))
                )
            )
        )
    return types.Content(parts=parts)

def chatbot_injury_assistance(user_query, image_path):
    client = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
    prompt = f"""You are a medical image analysis assistant...
Analyze the injury image and extract:
1. Injury type
2. Body location
3. severity
4. caused_by
5. species
Return ONLY a JSON object with keys:
- injury_type
- body_location
- severity
- species
- caused_by

User's concern: {user_query}
"""

    content = create_content(prompt, image_path)
    result = client.invoke(content)
    return result
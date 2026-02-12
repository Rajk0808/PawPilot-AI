import os
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from AI_Model.src.prompt_engineering.food_model_prompts import get_food_vision_prompt, route_food_query
from langchain_core.messages import HumanMessage



def decode_image(image):
    image.seek(0)
    return base64.b64encode(image.read()).decode('utf-8')

def create_content(prompt, image):
    content: list = [{"type": "text", "text": str(prompt)}]
    for img in image:
        img.seek(0)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(img.read()).decode('utf-8')}"
                }
            }
        )
    return content


def chatbot_food_analyzer(user_query, image):
    client = ChatGoogleGenerativeAI(model='gemini-3-flash-preview')
    route_info = route_food_query(user_query)
    prompt = get_food_vision_prompt(route_info.get('vision_context', 'standard'), route_info.get('species', 'unknown'))
    content = create_content(prompt, image)
    message = HumanMessage(content=content)
    result = client.invoke([message])

    return result
from PIL import Image
import base64
class LoadImages:
    def image_loader(self, strategy,image_paths):
        images = []
        if strategy == "PIL":
            for img in image_paths:
                if isinstance(img, Image.Image):
                    images.append(img.convert("RGB"))
                else:
                    images.append(Image.open(img).convert("RGB"))
            return images

        elif strategy == 'Base64':

            for img_path in image_paths:
                with open(img_path, "rb") as img_file:
                    b64_string = base64.b64encode(img_file.read()).decode('utf-8')
                    images.append(b64_string)
        elif strategy == 'BytesIO':
            import io
            for img_bytes in image_paths:
                if isinstance(img_bytes, bytes):
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    images.append(img)
                else:
                    raise ValueError("Expected bytes input for BytesIO strategy.")
                
        return images
        
    
    def image_to_data_url(self, image_path):
        result = []
        for img in image_path:
            with open(img, "rb") as img_file:
                b64_string = base64.b64encode(img_file.read()).decode('utf-8')
                result.append(f"data:image/jpeg;base64,{b64_string}")
        return result

class MessageLoader:
    def LoadMessages(self,model, query, images):
        base_message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Describe this image in detail and also detect the {query}"},
                ],
            }
        ]
        if model == "gpt-4o-mini":
            for img in images:
                base_message[0]["content"].append({"type": "image", "image": img})
            return base_message
        elif model == "nvidia/nemotron-nano-12b-v2-vl:free" or model == "allenai/molmo-2-8b:free":
            for img in images:
                base_message[0]["content"].append({"type": "image_url", "image_url": img})
        return base_message
    

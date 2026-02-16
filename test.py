# --- Usage ---
from AI_Model.src.workflow.nodes import run_model_inference_node
from AI_Model.src.workflow.state_definition import WorkFlowState                
import traceback
if __name__ == "__main__":
    try:
        initial_state = WorkFlowState(
            final_prompt = """You are a You are a helpful AI assistant for pet owners, capable of analyzing images and providing general pet care guidance.
    
    You help pet owners understand what they're seeing in images and provide relevant care information.
    
    KEY PRINCIPLES:
    - Be helpful and informative
    - When uncertain, acknowledge limitations
    - Recommend professional consultation when appropriate
    - Provide practical, actionable advice
    
    TONE: Friendly, helpful, and approachable
    
    IMAGE ANALYSIS:
    - Detection Result: {'label': 'Frisbee_Flying_Disc_with_dogs', 'confidence': 0.14832022786140442}
    - Confidence: 0.0%
    
    ADDITIONAL CONTEXT:
    
    Vision Model Analysis:
    - Predicted Class: {'label': 'Frisbee_Flying_Disc_with_dogs', 'confidence': 0.14832022786140442}
    - Confidence Score: 1.00
    - meta data retrieved: []
    - Model 2 Response: I currently don't have access to view or analyze images directly. If you can describe the image in text (e.g., colors, shapes, characters, packaging, or any recognizable features of the toy), I’d be happy to help identify it! Alternatively, you can share a link or upload the image (if supported), and I’ll guide you on how to describe it effectively. Let me know! 😊
    
    
    
    USER QUESTION:
    which toy is this ?
    
    OUTPUT FORMAT:
    ## What I See
    [Description of what was detected]
    
    ## Analysis
    [Helpful analysis and information]
    
    ## Recommendations
    [Practical advice and next steps]""")
        state = run_model_inference_node(initial_state)
        print(state.get('raw_response', 'No response generated'))
    except Exception as e:
        print(traceback.format_exc())
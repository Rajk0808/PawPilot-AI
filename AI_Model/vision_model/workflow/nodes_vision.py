import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from AI_Model.vision_model.workflow.state_definition_vision import VisionWorkFlowState
from time import time 
from AI_Model.vision_model.rag_vision.retriever_vision import retrieve_docs
from AI_Model.vision_model.model.image_detect_model import call_nvdia 
from AI_Model.src.utils.exceptions import CustomException
from AI_Model.vision_model.utils.keyword_extractor import KeyWordExtractor
import logging

logger = logging.getLogger(__name__)

def input_processing_node(state: VisionWorkFlowState) -> VisionWorkFlowState:
    """
    Process user input from the state dictionary.
    
    Args:
        state: Current workflow state as a VisionWorkFlowState.
    """
    logger.info('Processing user input...')

    try:
        query = state.get("query", "")
        image = state.get("image", None)
        user_id = state.get("user_id", "unknown")
        session_id = state.get("session_id", "unknown")
        if image is not None:
            state['image'] = image
        state["query"] = query
        logger.info(f'Processing query from user {user_id} in session {session_id}')

        if not query or query.strip() == "":
            logger.warning("Invalid/empty query provided")
            state['strategy'] = 'all_models'
            return state
        
        if len(query) > 2000:
            logger.warning(f"Query too long ({len(query)} chars), truncating...")
            state["error"] = ["Query too long"]
            state["query"] = query[:2000]
        
        # Initialize metadata
        state["start_time"] = time()
        
        return state

    except Exception as e: 
        raise CustomException(e, sys)
    

def decision_router_node(state: VisionWorkFlowState) -> VisionWorkFlowState:
    """
    Decide on the model strategy based on the state.
    
    Args:
        state: Current workflow state as a dictionary.
    """
    logger.info('Deciding on model strategy...')

    try:
        query = state.get("query", "")
        extractor = KeyWordExtractor()
        state['strategy'] = extractor.select_strategy(query)        
        return state

    except Exception as e: 
        raise CustomException(e, sys)

def model_call_node(state: VisionWorkFlowState) -> VisionWorkFlowState:
    """
    Call the appropriate model based on the strategy in the state.
    
    Args:
        state: Current workflow state as a dictionary.
    """
    logger.info('Calling the appropriate model...')

    try:
        strategy = state.get("strategy", "")
        image = state.get("image")
        
        if strategy is None:
            strategy = ""
        
        # Check if image is provided
        if image is None:
            logger.error("No image provided in state")
            state["predicted_class"] = "unknown"
            state["confidence_score"] = 0.0
            if "error" not in state or state.get("error") is None:
                state["error"] = []
            state["error"].append("No image provided")
            return state
        
        if strategy == "diseases_classifier":
            from AI_Model.vision_model.model.diseases_model_prediction import predict, load_model
            model, preprocess, classifier, id2label, device = load_model()
            results = predict(image, preprocess, model, classifier, id2label, device)
            
            state["predicted_class"] = results.get("label", "unknown") if isinstance(results, dict) else "unknown"
            state["confidence_score"] = results.get("confidence", 0.0) if isinstance(results, dict) else 0.0

            return state
            
        elif strategy == "toy_classifier":
            from AI_Model.vision_model.model.toy_model_prediction import predict_toy, load_model_toy
            model, preprocess, classifier, id2label, device = load_model_toy()
            results = predict_toy(image, preprocess, model, classifier, id2label, device)
            
            # Handle different return types
            if results is None:
                state["predicted_class"] = "unknown"
                state["confidence_score"] = 0.0
            elif isinstance(results, str):
                # If predict returns a string directly
                state["predicted_class"] = results
                state["confidence_score"] = 1.0
            elif isinstance(results, list) and len(results) > 0:
                # If predict returns a list of dicts
                if isinstance(results[0], dict):
                    state["predicted_class"] = results[0].get("label", "unknown")
                    state["confidence_score"] = results[0].get("confidence", 0.0)
                else:
                    # List of strings
                    state["predicted_class"] = str(results[0])
                    state["confidence_score"] = 1.0
            elif isinstance(results, dict):
                # If predict returns a single dict
                state["predicted_class"] = results.get("label", "unknown")
                state["confidence_score"] = results.get("confidence", 0.0)
            else:
                state["predicted_class"] = "unknown"
                state["confidence_score"] = 0.0
        
        elif strategy == "emotion_detection":
            from AI_Model.vision_model.model.emotion_detection import chatbot_emotion_detection
            user_query = state.get("query", "")
            images = [image]  # Assuming single image for emotion detection
            reply = chatbot_emotion_detection(user_query, images)
            state["final_output"] = reply
            state["predicted_class"] = "emotion_detected"
            state["confidence_score"] = 1.0  # Assuming full confidence for text response
        
        elif strategy == "full-body-scan":
            from AI_Model.vision_model.model.full_body_scan import chatbot_full_body_scan
            user_query = state.get("query", "")
            images = [image]  # Assuming single image for full body scan
            reply = chatbot_full_body_scan(user_query, images)
            state["final_output"] = reply
            state["predicted_class"] = "full_body_scan_completed"
            state["confidence_score"] = 1.0  # Assuming full confidence for text response
        
        elif strategy == "parasite-detection":
            from AI_Model.vision_model.model.parasites_detection import predict
            

        else:
            logger.warning(f"Unknown strategy: {strategy}")
            state["predicted_class"] = "unknown"
            state["confidence_score"] = 0.0
        return state

    except Exception as e: 
        raise CustomException(e, sys)
    

def second_model_node(state: VisionWorkFlowState) -> VisionWorkFlowState:
    """
    Optionally call a second model based on confidence score.
    
    Args:
        state: Current workflow state as a dictionary.
    """
    logger.info('Evaluating need for second model call...')

    try:
        if state.get("strategy") != "emotion_detection":
            state['raw_model2_response'] = call_nvdia(state.get('image'))            
        return state

    except Exception as e: 
        raise CustomException(e, sys)


def retrieval_node(state: VisionWorkFlowState) -> VisionWorkFlowState:
    """
    Placeholder retrieval node for vision model workflow.
    
    Args:
        state: Current workflow state as a dictionary.
    """
    logger.info('Retrieval node - no operation for vision model.')

    try:
        if state.get("strategy") != "emotion_detection" and state.get("strategy") != "full-body-scan":
            # Currently no retrieval logic for vision model
            class_name = state.get("predicted_class", "unknown")
            strategy = state.get("strategy", "default")
            if strategy == "diseases_classifier":
                host_name = "https://dog-disease-6i6jnuf.svc.aped-4627-b74a.pinecone.io"
            elif strategy == "toy_classifier":
                host_name = "https://toy-detection-6i6jnuf.svc.aped-4627-b74a.pinecone.io"
            elif strategy == "parasite-detection":
                host_name = "https://parasite-detection-6i6jnuf.svc.aped-4627-b74a.pinecone.io"
            # Convert list to dict format if needed
            retrieved_result = retrieve_docs(class_name, host_name)
            state["retrieved_docs"] = retrieved_result if isinstance(retrieved_result, dict) else {}
        state['end_time'] = time()
        start_time = state.get('start_time', 0)
        state['inference_time'] = state['end_time'] - start_time
        return state

    except Exception as e: 
        raise CustomException(e, sys)
    
if __name__ == "__main__":
    # Example usage
    state = VisionWorkFlowState()
    state['query'] = "chunkied"
    state = decision_router_node(state)
    print(f"Selected strategy: {state.get('strategy')}")
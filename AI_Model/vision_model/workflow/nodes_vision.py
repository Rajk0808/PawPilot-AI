import sys
from AI_Model.vision_model.workflow.state_definition_vision import VisionWorkFlowState
from time import time 
from AI_Model.vision_model.rag_vision.retriever_vision import doc_retriver,MetaDataStore
from AI_Model.vision_model.model.image_detect_model import call_nvdia 
from AI_Model.src.utils.exceptions import CustomException
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
        diseases_keywords = set('diseases illness ringworm mange cancer tumor cyst infection hot spot dermatitis yeast fleas ticks mites rash redness hair loss alopecia bald spots swelling bumps lumps scabs bleeding pus oozing crusty flaky skin dandruff inflammation lesion wound injury cut scratch bite mark itchy scratching licking painful hurting vet emergency contagious dangerous treatment diagnosis'.split())
        toy_keywords = set('toy play fetch chew ball frisbee tug rope squeaky bone treat reward fun exercise training obedience agility socialization companionship toy bone ball frisbee rope plush squeaky stuffed animal puzzle scratcher wand tunnel rubber plastic nylon latex fabric wood antler rawhide silicone splinter sharp edges swallow choke stuck blockage pieces stuffing break snap ingest digestive teeth gums chew destroy shred rip play fetch tug durability brand make model material non-toxic bpa-free heavy duty indestructible'.split())
        # Simple heuristic for strategy decision

        if any(keyword in query.lower() for keyword in diseases_keywords) or "condition" in query.lower():
            state["strategy"] = "diseases_classifier"
            state['model_to_use'] = 'diseases_classifier'
            logger.info("Selected strategy: diseases_classifier")
        elif any(keyword in query.lower() for keyword in toy_keywords) or "toy" in query.lower():
            state["strategy"] = "toy_classifier"
            state['model_to_use'] = 'toy_classifier'
            logger.info("Selected strategy: toy_classifier")
        else:
            state["strategy"] = "default"
            logger.info("Selected strategy: default")
        
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
                    state["predicted_class"] = "unknown"  #results[0].get("label", "unknown")
                    state["confidence_score"] =  0.0 #results[0].get("confidence", 0.0)
                else:
                    # List of strings
                    state["predicted_class"] = str(results[0])
                    state["confidence_score"] = 1.0
            elif isinstance(results, dict):
                # If predict returns a single dict
                state["predicted_class"] = "unknown"  #results.get("label", "unknown")
                state["confidence_score"] = 0.0 #results.get("confidence", 0.0)
            else:
                state["predicted_class"] = "unknown"
                state["confidence_score"] = 0.0
            
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
        
        else:
            logger.warning(f"Unknown strategy: {strategy}")
            state["predicted_class"] = "unknown"
            state["confidence_score"] = 0.0
        
        logger.info(f"Prediction result: {state['predicted_class']} (confidence: {state['confidence_score']:.2f})")
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
        if state.get("strategy") != "emotion_detection":
            # Currently no retrieval logic for vision model
            class_name = state.get("predicted_class", "unknown")
            strategy = state.get("strategy", "default")
            retriver = doc_retriver()
            metadataclient = MetaDataStore()
            client = metadataclient.get_client()
            collection_name = 'DogDisease' if strategy == 'diseases_classifier' else 'ToyDetection'
            property_name = 'disease_name' if strategy == 'diseases_classifier' else 'toy_name'
            docs = retriver.retriver(client, query=class_name, collection_name=collection_name, property_name=property_name)
            # Convert list to dict format if needed
            state["retrieved_docs"] = {"documents": docs} if isinstance(docs, list) else docs
        state['end_time'] = time()
        start_time = state.get('start_time', 0)
        state['inference_time'] = state['end_time'] - start_time
        return state

    except Exception as e: 
        raise CustomException(e, sys)
    

from typing import Dict, Optional, List
import json
from pathlib import Path
import logging
from .food_model_prompts import (
    get_response_prompt as get_food_response_prompt,
    route_food_query,
    FoodQueryRouter
)
from .Injury_model_prompt import get_prompt_for_context, get_generation_prompt_for_context, detect_injury_context_from_text  
logger = logging.getLogger(__name__)

class PawPilotPromptBuilder:
    """Build specialized prompts for PawPilot AI modules"""
    
    def __init__(self):
        self.templates_dir = Path("AI_Model/src/prompt_engineering/templates")
        self.load_all_templates()
    
    def load_all_templates(self):
        """Load all PawPilot-specific templates"""
        self.system_prompts = self._load_json("system_prompts.json")
        self.vision_prompts = self._load_json("vision_prompts.json")
        self.audio_prompts = self._load_json("audio_prompt.json")  # Note: file is audio_prompt.json not audio_prompts.json
        self.few_shot = self._load_json("few_shot_examples.json")
        self.rag_templates = self._load_json("rag_context_templates.json")
    
    def _load_json(self, filename: str) -> Dict:
        """Load JSON file with error handling for missing files"""
        path = self.templates_dir / filename
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Template file not found: {path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {path}: {e}")
            return {}
    
    # ==========================================
    # SKIN DIAGNOSIS PROMPT
    # ==========================================
    
    def build_skin_diagnosis_prompt(
        self,
        pet_profile: Dict,
        symptom_description: str,
        rag_context: str
    ) -> str:
        """
        Build prompt for analyzing pet skin conditions
        
        Args:
            pet_profile: {"name": "Max", "breed": "Golden", "age": 4, "allergies": [...]}
            symptom_description: Description of what's visible in image
            rag_context: Retrieved relevant conditions from RAG database
        """
        
        system = self.system_prompts["skin_health_diagnostic"]
        
        prompt = f"""You are a {system['role']}
        
        {system['context']}
        
        KEY PRINCIPLES:
        {chr(10).join(f"- {p}" for p in system['key_principles'])}
        
        PET INFORMATION:
        - Name: {pet_profile.get('name', 'Unknown')}
        - Breed: {pet_profile.get('breed', 'Unknown')}
        - Age: {pet_profile.get('age', 'Unknown')} years
        - Allergies: {', '.join(pet_profile.get('allergies', ['None known']))}
        - Medical History: {pet_profile.get('medical_history', 'None reported')}
        
        KNOWLEDGE BASE REFERENCE (Similar conditions from RAG):
        {rag_context}
        
        SYMPTOM ANALYSIS:
        {symptom_description}
        
        ANALYSIS REQUIRED:
        1. Examine described symptoms
        2. Cross-reference with RAG knowledge base
        3. Consider pet's health history
        4. Assess severity and urgency
        5. Provide first aid steps
        6. Recommend when to see vet
        
        OUTPUT FORMAT:
        ## Observations
        [What you see in the symptoms]
        
        ## Possible Conditions
        - [Condition 1] - Likelihood: [High/Medium/Low]
        - [Condition 2] - Likelihood: [High/Medium/Low]
        
        ## Severity Level
        [Low/Medium/High/Emergency]
        
        ## Urgency for Vet Visit
        [Within 1 week / 48-72 hours / 24 hours / IMMEDIATE]
        
        ## First Aid Steps
        1. [Action]
        2. [Action]
        3. [Action]
        
        ## Monitoring
        What to watch for and when to escalate
        
        ## Important Notes
        Always phrase as "possible conditions" not diagnosis. When uncertain, recommend vet visit."""
                
        return prompt
            
    # ==========================================
    # EMOTION DETECTION PROMPT
    # ==========================================
    
    def build_emotion_detection_prompt(
        self,
        image_features: str,
        audio_analysis: str,
        pet_profile: Dict,
        rag_emotion_data: str
    ) -> str:
        """Build prompt for EmoDetect emotion analysis"""
        
        system = self.system_prompts["voice_emotion_translator"]
        
        prompt = f"""You are an {system['role']}
        
        {system['context']}
        
        EMOTION DETECTION FRAMEWORK (From EmoDetect RAG):
        {rag_emotion_data}
        
        PET CONTEXT:
        - Name: {pet_profile.get('name')}
        - Breed: {pet_profile.get('breed')}
        - Age: {pet_profile.get('age')} years
        - Personality: {pet_profile.get('personality', 'Unknown')}
        - Recent Events: {pet_profile.get('recent_events', 'None reported')}
        
        VISUAL ANALYSIS:
        {image_features}
        
        AUDIO ANALYSIS:
        {audio_analysis}
        
        DETECTION TASK:
        1. Identify body language indicators (using EmoDetect framework)
        2. Analyze audio patterns and vocalizations
        3. Consider pet's personality and recent context
        4. Cross-reference with EmoDetect emotion taxonomy
        5. Assess confidence level of emotion detection
        6. Recommend appropriate actions
        
        OUTPUT FORMAT:
        ## Primary Emotion
        [Emotion from EmoDetect list]
        
        ## Confidence Level
        High / Medium / Low
        
        ## Key Indicators Observed
        - Body Language: [indicators]
        - Vocalizations: [audio patterns]
        - Context Clues: [situational factors]
        
        ## Root Cause Analysis
        [What triggered this emotion]
        
        ## Recommended Actions
        1. [What to do]
        2. [How to help]
        3. [When to seek help]
        
        ## Important Notes
        Be specific about which body parts indicate which emotions using EmoDetect framework."""
        
        return prompt
    
    # ==========================================
    # EMERGENCY RESPONSE PROMPT
    # ==========================================
    
    def build_emergency_prompt(
        self,
        emergency_type: str,
        symptoms: str,
        pet_profile: Dict,
        rag_emergency_protocols: str
    ) -> str:
        """Build CRITICAL CARE prompt for emergencies"""
        
        system = self.system_prompts["emergency_assistant"]
        
        prompt = f"""You are a {system['role']}

        {system['context']}
        
        KEY PRINCIPLES FOR LIFE-SAVING RESPONSES:
        {chr(10).join(f"- {p}" for p in system['key_principles'])}
        
        PET INFORMATION:
        - Age: {pet_profile.get('age')} years
        - Weight: {pet_profile.get('weight')} kg
        - Medical Conditions: {pet_profile.get('medical_conditions', 'None reported')}
        - Current Medications: {pet_profile.get('medications', 'None')}
        
        EMERGENCY TYPE: {emergency_type}
        
        SYMPTOMS REPORTED:
        {symptoms}
        
        EMERGENCY PROTOCOLS (From RAG):
        {rag_emergency_protocols}
        
        CRITICAL RESPONSE REQUIRED:
        ⚠️ PROVIDE NUMBERED STEPS ONLY - NO PARAGRAPHS
        ⚠️ START WITH SEVERITY ASSESSMENT
        ⚠️ INCLUDE WHAT NOT TO DO
        ⚠️ SPECIFY TIME WINDOWS
        
        OUTPUT FORMAT (STRICT):
        ## 🚨 SEVERITY LEVEL
        [LIFE-THREATENING / SERIOUS / URGENT]
        
        ## ⏱️ TIME CRITICAL WINDOW
        [How much time available before escalation needed]
        
        ## 🔴 IMMEDIATE ACTIONS (IN ORDER):
        1. [Action - be specific]
        2. [Action - be specific]
        3. [Action - be specific]
        4. [Action - be specific]
        
        ## ❌ WHAT NOT TO DO:
        - Do NOT [something dangerous]
        - Do NOT [something dangerous]
        - Do NOT [something dangerous]
        
        ## 📞 VET URGENCY
        [CALL IMMEDIATELY / Go to ER now / Vet within 1 hour]
        
        ## 🩺 EQUIPMENT NEEDED
        [List specific items needed]
        
        ## 👁️ WARNING SIGNS FOR ESCALATION
        [When to stop and go to vet immediately]"""
                
        return prompt
    
    # ==========================================
    # PRODUCT SAFETY PROMPT
    # ==========================================
    
    def build_product_analysis_prompt(
        self,
        product_info: Dict,
        pet_profile: Dict,
        rag_safety_database: str
    ) -> str:
        """Build prompt for product safety evaluation"""
        
        system = self.system_prompts["product_safety_evaluator"]
        
        prompt = f"""You are a {system['role']}
        
        {system['context']}
        
        SAFETY DATABASE REFERENCE:
        {rag_safety_database}
        
        PET PROFILE:
        - Species: {pet_profile.get('species')}
        - Age: {pet_profile.get('age')} years
        - Weight: {pet_profile.get('weight')} kg
        - Known Allergies: {', '.join(pet_profile.get('allergies', ['None']))}
        - Health Conditions: {pet_profile.get('health_conditions', 'None reported')}
        
        PRODUCT TO EVALUATE:
        - Name: {product_info.get('name')}
        - Type: {product_info.get('type')} (food/treat/toy/supplement)
        - Ingredients: {product_info.get('ingredients')}
        - Price: {product_info.get('price', 'Unknown')}
        
        EVALUATION STEPS:
        1. Check each ingredient against safety database
        2. Flag any toxic substances
        3. Assess portion/size appropriateness for pet
        4. Check for allergen risks specific to this pet
        5. Evaluate nutritional content
        6. Compare price vs value
        7. Suggest alternatives if needed
        
        OUTPUT FORMAT:
        ## 🏷️ Product Name & Type
        [Info]
        
        ## ✅ Safety Assessment
        Safe / Caution / Not Safe
        
        ## 🚨 Toxic Ingredients Found
        [List any toxic ingredients, or "None found"]
        
        ## ⚠️ Allergen Concerns
        [Any concerns for this specific pet]
        
        ## 📊 Safety Score
        [1-10 scale with explanation]
        
        ## 💡 Better Alternatives
        [Healthier/safer options]
        
        ## 📏 Appropriate Portion Size
        [Specific amount for this pet's weight]
        
        ## 💰 Value Assessment
        [Is it worth the price?]
        
        ## 🎯 Final Recommendation
        [Clear recommendation for this pet]"""
        
        return prompt
    
    # ==========================================
    # VISION - TOY CLASSIFIER PROMPT
    # ==========================================
    
    def build_toy_classifier_prompt(
        self,
        predicted_class: str,
        confidence_score: float,
        user_query: str,
        rag_context: str,
        pet_profile: Optional[Dict] = None
    ) -> str:
        """
        Build prompt for toy classification and safety analysis
        
        Args:
            predicted_class: The classified toy type from vision model
            confidence_score: Model confidence (0-1)
            user_query: User's question about the toy
            rag_context: Retrieved toy information from RAG
            pet_profile: Optional pet info for personalized recommendations
        """
        
        vision_config = self.vision_prompts.get("toy_classifier", {})
        system = vision_config.get("system_prompt", {})
        template = vision_config.get("response_template", {})
        thresholds = self.vision_prompts.get("confidence_thresholds", {})
        
        # Determine confidence level
        if confidence_score >= thresholds.get("high_confidence", 0.85):
            confidence_level = "High"
        elif confidence_score >= thresholds.get("medium_confidence", 0.65):
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        # Build pet context if available
        pet_context = ""
        if pet_profile:
            pet_context = f"""
PET INFORMATION:
- Name: {pet_profile.get('name', 'Unknown')}
- Species: {pet_profile.get('species', 'Unknown')}
- Breed: {pet_profile.get('breed', 'Unknown')}
- Age: {pet_profile.get('age', 'Unknown')} years
- Size: {pet_profile.get('size', 'Unknown')}
- Chewing Strength: {pet_profile.get('chewing_strength', 'Unknown')}
"""
        
        prompt = f"""You are a {system.get('role', 'pet product advisor')}

{system.get('context', '')}

KEY PRINCIPLES:
{chr(10).join(f"- {p}" for p in system.get('key_principles', []))}

TONE: {system.get('tone', 'Friendly and helpful')}

IMAGE ANALYSIS RESULT:
- Identified Product: {predicted_class}
- Confidence: {confidence_score:.1%} ({confidence_level})
{pet_context}
KNOWLEDGE BASE REFERENCE:
{rag_context if rag_context else "No additional product information available."}

USER QUESTION:
{user_query}

OUTPUT FORMAT:
## 🧸 Product Identification
[Identified toy type and confidence]

## ✅ Safety Assessment
[Safety evaluation for different pet types/sizes]

## 🐕 Best Suited For
[Which pets this toy is appropriate for]

## 🎮 Play Tips
[How to use this toy effectively]

## 🧹 Care & Maintenance
[How to clean and when to replace]

## ⚠️ Supervision Recommendation
[Always end with supervision advice]

CONSTRAINTS:
- Maximum {template.get('constraints', {}).get('max_words', 250)} words
- Be specific about safety concerns
- Include size/breed recommendations"""
        
        return prompt
    
    # ==========================================
    # VISION - DISEASES CLASSIFIER PROMPT
    # ==========================================
    
    def build_diseases_classifier_prompt(
        self,
        predicted_class: str,
        confidence_score: float,
        user_query: str,
        rag_context: str,
        pet_profile: Optional[Dict] = None
    ) -> str:
        """
        Build prompt for pet skin condition/disease analysis
        
        Args:
            predicted_class: The classified condition from vision model
            confidence_score: Model confidence (0-1)
            user_query: User's question about the condition
            rag_context: Retrieved medical information from RAG
            pet_profile: Optional pet info for personalized assessment
        """
        
        vision_config = self.vision_prompts.get("diseases_classifier", {})
        system = vision_config.get("system_prompt", {})
        template = vision_config.get("response_template", {})
        thresholds = self.vision_prompts.get("confidence_thresholds", {})
        emergency_triggers = vision_config.get("emergency_triggers", [])
        
        # Determine confidence level
        if confidence_score >= thresholds.get("high_confidence", 0.85):
            confidence_level = "High"
        elif confidence_score >= thresholds.get("medium_confidence", 0.65):
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        # Check for emergency keywords in query
        is_emergency = any(trigger in user_query.lower() for trigger in emergency_triggers)
        
        # Build pet context if available
        pet_context = ""
        if pet_profile:
            pet_context = f"""
PET INFORMATION:
- Name: {pet_profile.get('name', 'Unknown')}
- Species: {pet_profile.get('species', 'Unknown')}
- Breed: {pet_profile.get('breed', 'Unknown')}
- Age: {pet_profile.get('age', 'Unknown')} years
- Weight: {pet_profile.get('weight', 'Unknown')} kg
- Known Allergies: {', '.join(pet_profile.get('allergies', ['None known']))}
- Medical History: {pet_profile.get('medical_history', 'None reported')}
"""
        
        emergency_notice = ""
        if is_emergency:
            emergency_notice = """
🚨 EMERGENCY INDICATORS DETECTED - PRIORITIZE IMMEDIATE GUIDANCE 🚨
"""
        
        prompt = f"""You are a {system.get('role', 'veterinary AI assistant')}

{system.get('context', '')}

KEY PRINCIPLES:
{chr(10).join(f"- {p}" for p in system.get('key_principles', []))}

TONE: {system.get('tone', 'Professional and caring')}
{emergency_notice}
IMAGE ANALYSIS RESULT:
- Possible Condition Detected: {predicted_class}
- Model Confidence: {confidence_score:.1%} ({confidence_level})
{pet_context}
MEDICAL KNOWLEDGE BASE REFERENCE:
{rag_context if rag_context else "No additional medical information available from knowledge base."}

USER CONCERN:
{user_query}

OUTPUT FORMAT:
## 👁️ Visual Observation
[What the image analysis detected]

## 🔍 Possible Condition
[Condition name with confidence - NEVER state as definitive diagnosis]

## ⚠️ Severity Level
[{' / '.join(template.get('severity_levels', ['Low', 'Medium', 'High', 'Emergency']))}]

## 🩹 Immediate Care Steps
1. [First aid action]
2. [Comfort measure]
3. [What to monitor]

## 🏥 Veterinary Recommendation
[When to see vet: Immediately / Within 24 hours / Within 48-72 hours / Within 1 week]

## ⚕️ Disclaimer
*This is an AI-assisted assessment and NOT a medical diagnosis. Always consult a licensed veterinarian for proper examination and treatment.*

CONSTRAINTS:
- Maximum {template.get('constraints', {}).get('max_words', 300)} words
- Always phrase as "possible condition" not "diagnosis"
- Always recommend veterinary consultation
- Include severity assessment
- Be empathetic but honest"""
        
        return prompt
    
    # ==========================================
    # VISION - DEFAULT/GENERAL PROMPT
    # ==========================================
    
    def build_vision_default_prompt(
        self,
        predicted_class: str,
        confidence_score: float,
        user_query: str,
        rag_context: str = ""
    ) -> str:
        """
        Build a general vision analysis prompt for unclassified images
        """
        
        vision_config = self.vision_prompts.get("default", {})
        system = vision_config.get("system_prompt", {})
        
        prompt = f"""You are a {system.get('role', 'helpful AI assistant for pet owners')}

{system.get('context', '')}

KEY PRINCIPLES:
{chr(10).join(f"- {p}" for p in system.get('key_principles', []))}

TONE: {system.get('tone', 'Friendly and helpful')}

IMAGE ANALYSIS:
- Detection Result: {predicted_class}
- Confidence: {confidence_score:.1%}

ADDITIONAL CONTEXT:
{rag_context if rag_context else "No additional context available."}

USER QUESTION:
{user_query}

OUTPUT FORMAT:
## What I See
[Description of what was detected]

## Analysis
[Helpful analysis and information]

## Recommendations
[Practical advice and next steps]"""
        
        return prompt
    
    # ==========================================
    # INJURY ASSISTANCE PROMPT
    # ==========================================
    def build_injury_assistance_prompt(self, user_query:str, rag_context: str="" ) -> str:
        """
        Build prompt for Injury analysis
        
        Args:
            injury_type: Type of the injure- cut, bite, scratch, etc
            injured_type : human, dog, cat
            severity_type: normal, moderate, high
            user_query: User's question about the condition
            rag_context: Retrieved medical information from RAG
            pet_profile: Optional pet info for personalized assessment
        """
        injury_context = detect_injury_context_from_text(user_query) #get the context about the injury that what is the injury actually
        specific_prompt = get_generation_prompt_for_context(injury_context) #generate the prompt for the particular injury

        final_prompt = specific_prompt + rag_context + user_query # make the final prompt combining all the text 
        return final_prompt
    
    # ==========================================
    # FOOD ANALYSIS MODEL PROMPT
    # ==========================================
    def build_food_analysis_model_prompt(self, user_query:str, rag_context:str="") -> str:
        """
        Build prompt for Food Safety analysis
        
        Args:
            user_query: User's question about food safety
            rag_context: Retrieved food safety information from RAG
    
        Returns:
        Formatted prompt string for food safety response generation
        """
        route_info =route_food_query(user_query) #get the intent of the query 
        specific_prompt = get_food_response_prompt(route_info['intent']) #get the specific prompt according to the intent
        final_prompt = str(specific_prompt) + "\n\n" + rag_context + "\n\n" + user_query #build the final prompt
        return final_prompt #return the full text

    # ==========================================
    # VISION - UNIFIED PROMPT BUILDER
    # ==========================================
    
    def build_vision_prompt(
        self,
        model_type: str,
        predicted_class: str,
        confidence_score: float,
        user_query: str,
        rag_context: str = "",
        pet_profile: Optional[Dict] = None
    ) -> str:
        """
        Unified vision prompt builder that routes to appropriate template
        
        Args:
            model_type: "toy_classifier", "diseases_classifier", or "default"
            predicted_class: Classification result from vision model
            confidence_score: Model confidence (0-1)
            user_query: User's question
            rag_context: Retrieved context from RAG
            pet_profile: Optional pet information
        """
        
        if model_type == "toy_classifier":
            return self.build_toy_classifier_prompt(
                predicted_class, confidence_score, user_query, rag_context, pet_profile
            )
        elif model_type == "diseases_classifier":
            return self.build_diseases_classifier_prompt(
                predicted_class, confidence_score, user_query, rag_context, pet_profile
            )
        elif model_type == 'injury_assistance':
            return self.build_injury_assistance_prompt(
                user_query, rag_context
            )
        elif model_type == 'pet_food_image_analysis':
            return self.build_food_analysis_model_prompt(user_query, rag_context
            )
        else:
            return self.build_vision_default_prompt(
                predicted_class, confidence_score, user_query, rag_context
            )


    # ==========================================
    # RAG-AWARE PROMPT BUILDER
    # ==========================================
    
    def build_rag_aware_prompt(
        self,
        module: str,
        user_query: Dict,
        pet_profile: Dict,
        rag_retrieved_data: str
    ) -> str:
        """
        Build prompt that intelligently uses RAG context
        
        The key to PawPilot: RAG data directly informs the prompt
        """
        
        if module == "skin_diagnosis":
            return self.build_skin_diagnosis_prompt(
                pet_profile, 
                user_query.get("symptom_description", ""),
                rag_retrieved_data
            )
        elif module == "emotion_detection":
            return self.build_emotion_detection_prompt(
                user_query["image_features"],
                user_query["audio_analysis"],
                pet_profile,
                rag_retrieved_data
            )
        elif module == "emergency":
            return self.build_emergency_prompt(
                user_query["emergency_type"],
                user_query["symptoms"],
                pet_profile,
                rag_retrieved_data
            )
        elif module == "product_safety":
            return self.build_product_analysis_prompt(
                user_query,
                pet_profile,
                rag_retrieved_data
            )
        elif module == "toy_classifier":
            return self.build_toy_classifier_prompt(
                user_query.get("predicted_class", "Unknown"),
                user_query.get("confidence_score", 0.0),
                user_query.get("query", ""),
                rag_retrieved_data,
                pet_profile
            )
        elif module == "diseases_classifier":
            return self.build_diseases_classifier_prompt(
                user_query.get("predicted_class", "Unknown"),
                user_query.get("confidence_score", 0.0),
                user_query.get("query", ""),
                rag_retrieved_data,
                pet_profile
            )
        elif module == "vision":
            # Generic vision module - auto-detect based on model_type
            return self.build_vision_prompt(
                user_query.get("model_type", "default"),
                user_query.get("predicted_class", "Unknown"),
                user_query.get("confidence_score", 0.0),
                user_query.get("query", ""),
                rag_retrieved_data,
                pet_profile
            )
        elif module == 'injury_assistance':
            return self.build_injury_assistance_prompt(
                user_query,
                rag_retrieved_data
            )
        elif module=='pet_food_image_analysis':
            return self.build_food_analysis_model_prompt(
                user_query, rag_retrieved_data
            )
        else:
            raise ValueError(f"Unknown module: {module}")
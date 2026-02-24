# Classifier agent for message classification

import json
import litellm
from typing import Dict
from .logging_config import get_logger
from ..constants.prompts import CLASSIFICATION_PROMPT

class ClassifierAgent:
    """Agent that classifies user messages into chit-chat or data queries, and determines if visualization is needed."""
    
    def __init__(self, api_key: str):
        self.logger = get_logger(__name__)
        self.api_key = api_key
        self.model = "gpt-4o-mini"
        
        # Set OpenAI API key for litellm
        litellm.api_key = api_key
        
    def classify_message(self, user_message: str) -> Dict:
        """
        Classify a user message to determine:
        1. Message type: chit_chat or data_query
        2. Whether visualization is needed
        
        Args:
            user_message: The user's input message
            
        Returns:
            dict with keys: message_type, needs_visualization, reasoning
        """
        self.logger.debug(f"Classifying message: {user_message[:100]}")
        
        try:
            # Format the prompt
            prompt = CLASSIFICATION_PROMPT.format(user_message=user_message)
            
            # Call OpenAI via litellm
            response = litellm.completion(
                model=f"openai/{self.model}",
                messages=[
                    {"role": "system", "content": "You are a helpful classification assistant. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            # Extract the response
            content = response.choices[0].message.content
            self.logger.debug(f"Classification response: {content}")
            
            # Parse JSON response
            classification = json.loads(content)
            
            # Validate response structure
            if "message_type" not in classification:
                raise ValueError("Missing 'message_type' in classification response")
            if "needs_visualization" not in classification:
                raise ValueError("Missing 'needs_visualization' in classification response")
            
            # Ensure message_type is valid
            if classification["message_type"] not in ["chit_chat", "data_query"]:
                self.logger.warning(f"Invalid message_type: {classification['message_type']}, defaulting to 'data_query'")
                classification["message_type"] = "data_query"
            
            self.logger.info(f"Classification: {classification['message_type']}, needs_visualization: {classification['needs_visualization']}")
            
            return classification
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse classification JSON: {e}")
            # Default to data_query if parsing fails
            return {
                "message_type": "data_query",
                "needs_visualization": False,
                "reasoning": "Failed to parse classification response"
            }
        except Exception as e:
            self.logger.error(f"Error in classify_message: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Default to data_query on error
            return {
                "message_type": "data_query",
                "needs_visualization": False,
                "reasoning": f"Error during classification: {str(e)}"
            }

from typing import Literal, Optional, Dict, Any
from langgraph.types import Command
from langgraph.graph import END
from functions.prereq import llm, backup_llm
from my_agent.user_state import UserProfile
from datetime import datetime
import logging
import time
from typing import List

# Set up logging to both terminal and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to terminal
        logging.FileHandler('app.log')  # Logs to file
    ]
)

logger = logging.getLogger(__name__)

class FallbackError(Exception):
    """Custom exception for fallback handling"""
    pass

class FallbackStrategy:
    """Class to manage fallback strategies and actions"""
    
    @staticmethod
    def handle_rate_limit(state: UserProfile, original_query: str) -> Dict[str, Any]:
        """Handle rate limit by trying backup LLM with exponential backoff"""
        max_retries = 3
        base_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                delay = base_delay * (2 ** attempt)
                logger.info(f"Rate limit hit, waiting {delay} seconds before retry {attempt + 1}")
                time.sleep(delay)
                
                # Try backup LLM
                response = backup_llm.invoke(original_query)
                return {
                    "success": True,
                    "response": response,
                    "used_backup": True
                }
            except Exception as e:
                logger.error(f"Backup LLM attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return {
                        "success": False,
                        "error": str(e)
                    }
    
    @staticmethod
    def handle_timeout(state: UserProfile, original_query: str) -> Dict[str, Any]:
        """Handle timeout by trying a simplified query or backup LLM"""
        try:
            # Try with a simplified query
            simplified_query = f"Please provide a brief response to: {original_query}"
            response = backup_llm.invoke(simplified_query)
            return {
                "success": True,
                "response": response,
                "used_simplified": True
            }
        except Exception as e:
            logger.error(f"Timeout recovery failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def handle_low_confidence(state: UserProfile, original_query: str) -> Dict[str, Any]:
        """Handle low confidence by trying different LLM configurations"""
        try:
            # Try with a different temperature setting
            response = llm.with_config({"temperature": 0.4}).invoke(original_query)
            return {
                "success": True,
                "response": response,
                "used_alternative_config": True
            }
        except Exception as e:
            logger.error(f"Low confidence recovery failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

def fallback_node(
    state: UserProfile,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    original_query: Optional[str] = None
) -> Command[Literal["onboarding_agent", "recommendation_agent", "__end__", "ask_gaido", "policy_info"]]:
    """
    Fallback node that handles various types of failures and unhandled queries.
    
    Args:
        state: Current user state
        error_type: Type of error that triggered the fallback
        error_message: Error message if available
        original_query: Original user query that caused the error
    """
    # Log the fallback event
    logger.info(f"Fallback triggered - Type: {error_type}, Message: {error_message}, Query: {original_query}")
    
    # Check if the error is related to interrupt - if so, ignore it and continue
    if error_type == "execution_error" and "interrupt" in str(error_message).lower():
        logger.info("Ignoring interrupt-related error as it's expected in normal operation")
        # Return to the node where the error occurred
        return Command(
            goto=state.current_workflow,  # Return to the current workflow node
            update={
                **state.model_dump(),
                "interaction_count": state.interaction_count + 1,
            }
        )
    
    # Initialize response and actions
    response = ""
    actions: List[str] = []
    recovery_attempted = False
    recovery_success = False
    
    # Handle different error types with specific strategies
    if error_type == "rate_limit":
        result = FallbackStrategy.handle_rate_limit(state, original_query)
        if result["success"]:
            response = result["response"]
            actions.append("Switched to backup LLM")
            recovery_attempted = True
            recovery_success = True
        else:
            response = """I'm currently experiencing high demand. 
Please try again in a few moments, or you can rephrase your question."""
            actions.append("Rate limit recovery failed")
    
    elif error_type == "timeout":
        result = FallbackStrategy.handle_timeout(state, original_query)
        if result["success"]:
            response = result["response"]
            actions.append("Used simplified query approach")
            recovery_attempted = True
            recovery_success = True
        else:
            response = """I'm taking longer than expected to process your request. 
Let's try again - could you please rephrase your question or try a different approach?"""
            actions.append("Timeout recovery failed")
    
    elif error_type == "low_confidence":
        result = FallbackStrategy.handle_low_confidence(state, original_query)
        if result["success"]:
            response = result["response"]
            actions.append("Adjusted LLM configuration")
            recovery_attempted = True
            recovery_success = True
        else:
            response = """I want to make sure I understand you correctly. Could you please:
1. Provide more details about what you're looking for
2. Or let me know which of these areas interests you:
   - Health insurance policies
   - Policy recommendations
   - Understanding coverage options"""
            actions.append("Low confidence recovery failed")
    
    elif error_type == "intent_failure":
        response = """I'm not quite sure I understand what you're looking for. Could you please:
1. Rephrase your question
2. Or let me know if you're looking for help with:
   - Finding a health insurance policy
   - Understanding your current policy
   - Getting recommendations based on your needs
   - Learning about our services"""
        actions.append("Requested user clarification")
    
    elif error_type == "execution_error":
        response = """I apologize, but I encountered an issue while processing your request. 
Let's try again - could you please rephrase your question or try a different approach?"""
        actions.append("Encountered execution error")
    
    elif error_type == "conflict":
        response = """I see multiple possible ways to help you. Could you clarify if you're looking for:
1. Policy recommendations
2. Onboarding assistance
3. General information about our services"""
        actions.append("Requested user clarification for conflicting intents")
    
    else:
        response = """I want to make sure I help you effectively. Could you please:
1. Rephrase your question
2. Or let me know if you're looking for help with:
   - Finding a health insurance policy
   - Understanding your current policy
   - Getting recommendations based on your needs"""
        actions.append("Generic fallback triggered")

    # Update state with fallback information
    state.fallback_triggered = True
    state.last_fallback_time = datetime.now().isoformat()
    state.fallback_type = error_type
    state.fallback_message = error_message
    state.fallback_actions = actions
    state.recovery_attempted = recovery_attempted
    state.recovery_success = recovery_success
    
    # Log recovery attempt results
    if recovery_attempted:
        logger.info(f"Recovery {'succeeded' if recovery_success else 'failed'} with actions: {actions}")
    
    # Return to the node where the error occurred
    return Command(
        goto=state.current_workflow,  # Return to the current workflow node
        update={
            **state.model_dump(),
            "user_intent_query": response,
            "interaction_count": state.interaction_count + 1,
        }
    )

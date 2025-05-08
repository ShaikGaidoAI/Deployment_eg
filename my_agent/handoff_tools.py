"""
Handoff Tools for Gaido Multi-Agent System

This module provides tools for handling agent handoffs in the Gaido multi-agent system.
It implements all possible handoffs between agents based on detected intents, routing logic,
and maintains context between transitions.
"""

from typing import Annotated, List, Dict, Optional, Any
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

# Import the state model to ensure type safety
from my_agent.user_state import UserProfile


def make_handoff_tool(*, agent_name: str):
    """
    Create a handoff tool that allows one agent to transfer control to another.
    
    Args:
        agent_name: The name of the agent to transfer control to.
        
    Returns:
        A tool function that creates a Command for handoff.
    """
    tool_name = f"transfer_to_{agent_name}"

    @tool(tool_name)
    def handoff_to_agent(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        reason: str = "",
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Ask another agent for help with a specific task.
        
        Args:
            state: The current state (injected automatically)
            tool_call_id: The current tool call ID (injected automatically) 
            reason: Reason for the handoff (optional)
            context: Additional context to pass to the target agent (optional)
        
        Returns:
            Command object for routing to the target agent
        """
        # Create tool message for continuity in conversation
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}. Reason: {reason if reason else 'Specialized assistance'}",
            "name": tool_name,
            "tool_call_id": tool_call_id,
        }
        
        # Convert state to UserProfile model dump if needed
        state_data = state.model_dump() if hasattr(state, "model_dump") else state
        
        # Combine with additional context if provided
        update_data = {**state_data}
        if context:
            update_data.update(context)
        
        # Add the handoff message to the state's messages
        if "messages" in update_data:
            update_data["messages"] = update_data["messages"] + [tool_message]
        else:
            update_data["messages"] = [tool_message]
        
        return Command(
            # Navigate to target agent node in the PARENT graph
            goto=agent_name,
            graph=Command.PARENT,
            # Pass full state with updated context
            update=update_data
        )

    return handoff_to_agent


# === ONBOARDING AGENT HANDOFF TOOLS ===

def get_onboarding_agent_tools():
    """
    Get all handoff tools for the onboarding agent.
    
    Returns:
        List of handoff tools available to the onboarding agent
    """
    return [
        make_handoff_tool(agent_name="recommendation_agent"),
        make_handoff_tool(agent_name="policy_info"),
        make_handoff_tool(agent_name="policy_comparison"),
        make_handoff_tool(agent_name="supervisor_node"),
        make_handoff_tool(agent_name="fallback_node")
    ]


# === RECOMMENDATION AGENT HANDOFF TOOLS ===

def get_recommendation_agent_tools():
    """
    Get all handoff tools for the recommendation agent.
    
    Returns:
        List of handoff tools available to the recommendation agent
    """
    return [
        make_handoff_tool(agent_name="onboarding_agent"),
        make_handoff_tool(agent_name="policy_info"),
        make_handoff_tool(agent_name="policy_comparison"),
        make_handoff_tool(agent_name="supervisor_node"),
        make_handoff_tool(agent_name="fallback_node")
    ]


# === POLICY INFO AGENT HANDOFF TOOLS ===

def get_policy_info_tools():
    """
    Get all handoff tools for the policy information agent.
    
    Returns:
        List of handoff tools available to the policy info agent
    """
    return [
        make_handoff_tool(agent_name="onboarding_agent"),
        make_handoff_tool(agent_name="recommendation_agent"),
        make_handoff_tool(agent_name="policy_comparison"),
        make_handoff_tool(agent_name="supervisor_node"),
        make_handoff_tool(agent_name="fallback_node")
    ]


# === POLICY COMPARISON AGENT HANDOFF TOOLS ===

def get_policy_comparison_tools():
    """
    Get all handoff tools for the policy comparison agent.
    
    Returns:
        List of handoff tools available to the policy comparison agent
    """
    return [
        make_handoff_tool(agent_name="onboarding_agent"),
        make_handoff_tool(agent_name="recommendation_agent"),
        make_handoff_tool(agent_name="policy_info"),
        make_handoff_tool(agent_name="supervisor_node"),
        make_handoff_tool(agent_name="fallback_node")
    ]


# === SUPERVISOR NODE HANDOFF TOOLS ===

def get_supervisor_tools():
    """
    Get all handoff tools for the supervisor node.
    
    Returns:
        List of handoff tools available to the supervisor node
    """
    return [
        make_handoff_tool(agent_name="onboarding_agent"),
        make_handoff_tool(agent_name="recommendation_agent"),
        make_handoff_tool(agent_name="policy_info"),
        make_handoff_tool(agent_name="policy_comparison"),
        make_handoff_tool(agent_name="fallback_node"),
        make_handoff_tool(agent_name="ask_gaido")
    ]


# === INTENT-BASED HANDOFF HELPERS ===

def create_intent_handoff_tool(intent_type: str):
    """
    Creates a tool that can detect a specific intent and hand off to the appropriate agent.
    
    Args:
        intent_type: The type of intent to detect
        
    Returns:
        A tool function that creates a Command for handoff based on intent
    """
    intent_map = {
        "policy_recommendation": "recommendation_agent",
        "policy_information": "policy_info",
        "policy_comparison": "policy_comparison",
        "onboarding": "onboarding_agent",
        "customer_support": "fallback_node",
        "general_query": "ask_gaido"
    }
    
    target_agent = intent_map.get(intent_type, "supervisor_node")
    tool_name = f"handle_{intent_type}_intent"
    
    @tool(tool_name)
    def intent_based_handoff(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        detected_intent: str = "",
        confidence: float = 0.0,
        extracted_entities: Optional[Dict[str, Any]] = None
    ):
        """
        Handle a specific intent detection and route to the appropriate agent.
        
        Args:
            state: The current state (injected automatically)
            tool_call_id: The current tool call ID (injected automatically)
            detected_intent: The detected intent (optional)
            confidence: Confidence score of the intent detection (optional)
            extracted_entities: Entities extracted from the user query (optional)
        
        Returns:
            Command object for routing to the appropriate agent
        """
        # Create tool message for continuity in conversation
        tool_message = {
            "role": "tool",
            "content": f"Detected {intent_type} intent with {confidence:.2f} confidence. Routing to appropriate handler.",
            "name": tool_name,
            "tool_call_id": tool_call_id,
        }
        
        # Convert state to UserProfile model dump if needed
        state_data = state.model_dump() if hasattr(state, "model_dump") else state
        
        # Update context with intent information
        context = {
            "detected_intent": detected_intent or intent_type,
            "intent_confidence": confidence,
            "current_workflow": intent_type,
            "interaction_count": state_data.get("interaction_count", 0) + 1
        }
        
        # Add extracted entities if provided
        if extracted_entities:
            context["extracted_entities"] = extracted_entities
        
        # Combine with state data
        update_data = {**state_data, **context}
        
        # Add the handoff message to messages
        if "messages" in update_data:
            update_data["messages"] = update_data["messages"] + [tool_message]
        else:
            update_data["messages"] = [tool_message]
        
        return Command(
            goto=target_agent,
            graph=Command.PARENT,
            update=update_data
        )
    
    return intent_based_handoff


def get_intent_detection_tools():
    """
    Get a list of all intent detection tools.
    
    Returns:
        List of intent detection tools
    """
    return [
        create_intent_handoff_tool("policy_recommendation"),
        create_intent_handoff_tool("policy_information"),
        create_intent_handoff_tool("policy_comparison"),
        create_intent_handoff_tool("onboarding"),
        create_intent_handoff_tool("customer_support"),
        create_intent_handoff_tool("general_query")
    ]


# === CONTEXT PRESERVATION TOOLS ===

@tool("save_context_and_handoff")
def save_context_and_handoff(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    target_agent: str,
    context_to_save: Dict[str, Any],
    reason: str = "Context transition"
):
    """
    Save specific context information and hand off to another agent.
    
    Args:
        state: The current state (injected automatically)
        tool_call_id: The current tool call ID (injected automatically)
        target_agent: The name of the agent to transfer control to
        context_to_save: Dictionary of context data to save before handoff
        reason: Reason for the handoff (optional)
    
    Returns:
        Command object for routing to the target agent with preserved context
    """
    # Create tool message for continuity in conversation
    tool_message = {
        "role": "tool",
        "content": f"Saving context and transferring to {target_agent}. Reason: {reason}",
        "name": "save_context_and_handoff",
        "tool_call_id": tool_call_id,
    }
    
    # Convert state to UserProfile model dump if needed
    state_data = state.model_dump() if hasattr(state, "model_dump") else state
    
    # Combine with context to save
    update_data = {**state_data}
    for key, value in context_to_save.items():
        update_data[key] = value
    
    # Add transfer metadata
    update_data["previous_agent"] = state_data.get("current_workflow", "unknown")
    update_data["handoff_reason"] = reason
    update_data["current_workflow"] = target_agent
    update_data["interaction_count"] = state_data.get("interaction_count", 0) + 1
    
    # Add the handoff message to messages
    if "messages" in update_data:
        update_data["messages"] = update_data["messages"] + [tool_message]
    else:
        update_data["messages"] = [tool_message]
    
    return Command(
        goto=target_agent,
        graph=Command.PARENT,
        update=update_data
    )


@tool("resume_previous_agent")
def resume_previous_agent(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    additional_context: Optional[Dict[str, Any]] = None
):
    """
    Resume the previous agent after completing a subtask.
    
    Args:
        state: The current state (injected automatically)
        tool_call_id: The current tool call ID (injected automatically)
        additional_context: Additional context to pass back (optional)
    
    Returns:
        Command object for routing back to the previous agent
    """
    # Convert state to UserProfile model dump if needed
    state_data = state.model_dump() if hasattr(state, "model_dump") else state
    
    # Get the previous agent from state or default to supervisor
    previous_agent = state_data.get("previous_agent", "supervisor_node")
    
    # Create tool message for continuity in conversation
    tool_message = {
        "role": "tool",
        "content": f"Resuming previous workflow with {previous_agent}.",
        "name": "resume_previous_agent",
        "tool_call_id": tool_call_id,
    }
    
    # Combine with additional context if provided
    update_data = {**state_data}
    if additional_context:
        update_data.update(additional_context)
    
    # Update workflow metadata
    update_data["current_workflow"] = previous_agent
    update_data["previous_agent"] = state_data.get("current_workflow", "unknown")
    update_data["interaction_count"] = state_data.get("interaction_count", 0) + 1
    
    # Add the handoff message to messages
    if "messages" in update_data:
        update_data["messages"] = update_data["messages"] + [tool_message]
    else:
        update_data["messages"] = [tool_message]
    
    return Command(
        goto=previous_agent,
        graph=Command.PARENT,
        update=update_data
    )


# === UTILITY FUNCTIONS ===

def get_all_handoff_tools():
    """
    Get all handoff tools available in the system.
    
    Returns:
        List of all handoff tools
    """
    intent_tools = get_intent_detection_tools()
    agent_specific_tools = [
        *get_onboarding_agent_tools(),
        *get_recommendation_agent_tools(),
        *get_policy_info_tools(),
        *get_policy_comparison_tools(),
        *get_supervisor_tools()
    ]
    
    context_tools = [
        save_context_and_handoff,
        resume_previous_agent
    ]
    
    # Use a set to remove duplicates (tools with the same name)
    unique_tools = {}
    for tool in [*intent_tools, *agent_specific_tools, *context_tools]:
        if hasattr(tool, 'name'):
            unique_tools[tool.name] = tool
        else:
            # For tools created with make_handoff_tool
            unique_tools[tool.__name__] = tool
    
    return list(unique_tools.values()) 
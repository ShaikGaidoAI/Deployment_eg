import os
import sys
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
from dotenv import load_dotenv
import random


# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
os.environ['LANGCHAIN_TRACING'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGSMITH_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = 'GAIDO-EXP'

from functions.prereq import llm, embeddings
# from functions.prereq import vectorstore
# from functions.fx import RAG_tool

from pydantic import BaseModel, Field
from typing import Optional, List
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END  
from langgraph.checkpoint.memory import MemorySaver

from my_agent.user_state import UserProfile
from my_agent.onb_hf import onboarding_workflow
# print('hi')
from typing import Literal



from typing import Literal
from langgraph.types import Command
from my_agent.user_state import UserProfile
from functions.prereq import llm
from langgraph.types import interrupt
from functions.prereq import QueryResponse
from functions.fx import RAG_tool, web_search_tool
from functions.fx import route_policy_query
from functions.fx import evaluate_rag_response

structured_llm = llm.with_structured_output(QueryResponse)


def policy_info_node(state: UserProfile) -> Command[Literal["onboarding_agent", "policy_info"]]:
    """
    Node that handles policy information requests using RAG and web search.
    Uses LLM to evaluate RAG responses and fall back to web search when needed.
    """
    print("\n" + "="*50)
    print("POLICY INFO NODE: Processing policy information request")
    print("="*50 + "\n")

    # If we don't have a user query yet, get one
    if not state.user_query:
        query_prompt = interrupt("What would you like to know about insurance policies?")
        return Command(
            goto="policy_info",
            update={
                **state.model_dump(),  # Include all existing state
                "user_query": query_prompt,
                "interaction_count": state.interaction_count + 1,
                "user_intent_query": None,
            }
        )

    # Route the query to the appropriate policy
    policy_name = route_policy_query(state.user_query)
    
    # First try RAG to get information from our database
    rag_answer = RAG_tool.invoke(state.user_query)
    
    # Evaluate if RAG response is sufficient
    if evaluate_rag_response(state.user_query, rag_answer):
        answer = rag_answer
        source = "our database"
    else:
        # If RAG response is insufficient, use web search
        answer = web_search_tool.invoke(state.user_query)
        source = "web search"
    
    # Present the information to the user
    user_response = interrupt(f"""
    Here's what I found about {policy_name} (from {source}):
    
    {answer}
    
    Would you like to:
    1. Ask another question about this policy
    2. Get information about a different policy
    3. Return to the main menu
    
    Please type your choice (1, 2, or 3):
    """)
    
    # Handle user response
    if user_response.strip() == "1":
        # Ask another question about the same policy
        return Command(
            goto="policy_info",
            update={
                **state.model_dump(),
                "user_query": None,  # Reset to get new query
                "interaction_count": state.interaction_count + 1
            }
        )
    elif user_response.strip() == "2":
        # Get information about a different policy
        return Command(
            goto="policy_info",
            update={
                **state.model_dump(),
                "user_query": None,  # Reset to get new query
                "interaction_count": state.interaction_count + 1
            }
        )
    else:
        # Return to main menu
        return Command(
            goto="onboarding_agent",
            update={
                **state.model_dump(),
                "user_query": None,  # Reset query
                "interaction_count": state.interaction_count + 1
            }
        ) 
    



def onboarding_agent_node(state: UserProfile)->Command[Literal["__end__", "policy_info"]]:
    """Wrapper for onboarding agent workflow"""
    onboarding_graph = onboarding_workflow.compile()
    # Run the onboarding workflow
    updated_state = onboarding_graph.invoke(state.model_dump())
    
    # Create a new state that includes all existing state plus updates
    return Command(
        goto=END,
        update={
            **state.model_dump(),  # Include all existing state
            **updated_state,  # Include all updates from onboarding
            "onboarding_complete": True,
            "current_workflow": "onboarding"
        }
    )

# Create the workflow graph
multi_agent_graph = StateGraph(UserProfile)


multi_agent_graph.add_node("onboarding_agent", onboarding_agent_node)
multi_agent_graph.add_node("policy_info", policy_info_node)

# Set the entry point - always start with the supervisor
multi_agent_graph.set_entry_point("onboarding_agent")

# Compile the graph with state persistence
graph_hf = multi_agent_graph.compile()
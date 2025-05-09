import os
import sys
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
from dotenv import load_dotenv
import random


# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
# os.environ['LANGCHAIN_TRACING'] = 'true'
# os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
# os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGSMITH_API_KEY")
# os.environ['LANGCHAIN_PROJECT'] = 'GAIDO-EXP'

from functions.prereq import llm, embeddings

# import json
# from supabase import create_client, Client

# url = os.getenv("SUPABASE_URL")
# key = os.getenv("SUPABASE_SERVICE_KEY")
# supabase: Client = create_client(url, key)

# import uuid
# from typing import Optional

# def update_state(state: UserProfile, id: Optional[str] = None):
#     record_id = str(id) if id else str(uuid.uuid4())
#     supabase.table("user_profiles").upsert({
#         "id": record_id,
#         "state": state.model_dump()
#     }).execute()

# def get_state(id: str):
#     # Retrieve the user profile by ID
#     response = supabase.table("user_profiles").select("state").eq("id", id).execute()
#     state_data = response.data[0]["state"]

#     # Deserialize into UserProfile
#     user_profile = UserProfile(**state_data)
#     return user_profile

from pydantic import BaseModel, Field
from typing import Optional, List
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END  
from langgraph.checkpoint.memory import MemorySaver

from my_agent.user_state import UserProfile
from my_agent.supervisor_node import supervisor_node
from my_agent.onboarding_agent import onboarding_workflow
# print('hi')
from my_agent.recommendation_agent import recommendation_workflow
from typing import Literal
from my_agent.policy_info_node import policy_info_node
from my_agent.policy_comparison_node import policy_comparison_node

MAX_ITERATIONS = 50  # Set a reasonable maximum number of iterations
TIMEOUT_SECONDS = 300

def onboarding_agent_node(state: UserProfile)->Command[Literal["recommendation_agent"]]:
    """Wrapper for onboarding agent workflow"""
    onboarding_graph = onboarding_workflow.compile()
    # Run the onboarding workflow
    updated_state = onboarding_graph.invoke(state.model_dump())
    
    # Create a new state that includes all existing state plus updates
    return Command(
        goto="recommendation_agent",
        update={
            **state.model_dump(),  # Include all existing state
            **updated_state,  # Include all updates from onboarding
            "onboarding_complete": True,
            "current_workflow": "onboarding"
        }
    )

def recommendation_agent_node(state: UserProfile)->Command[Literal["ask_gaido"]]:
    """Wrapper for recommendation agent workflow"""
    recommendation_graph = recommendation_workflow.compile()
    # Run the recommendation workflow
    updated_state = recommendation_graph.invoke(state.model_dump())
    
    # Create a new state that includes all existing state plus updates
    return Command(
        goto="ask_gaido",
        update={
            **state.model_dump(),  # Include all existing state
            **updated_state,  # Include all updates from recommendation
            "recommendation_complete": True,
            "current_workflow": "recommendation"
        }
    )

# Create the workflow graph
multi_agent_graph = StateGraph(UserProfile)  # Pass the class, not an instance

# Add all our nodes: supervisor, onboarding_agent, recommendation_agent
multi_agent_graph.add_node("ask_gaido", supervisor_node)
multi_agent_graph.add_node("onboarding_agent", onboarding_agent_node)
multi_agent_graph.add_node("recommendation_agent", recommendation_agent_node)
multi_agent_graph.add_node("policy_info", policy_info_node)
multi_agent_graph.add_node("policy_comparison", policy_comparison_node)

# Set the entry point - always start with the supervisor
multi_agent_graph.set_entry_point("ask_gaido")

# Compile the graph with state persistence
graph = multi_agent_graph.compile()
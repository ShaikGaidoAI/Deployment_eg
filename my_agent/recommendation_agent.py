import os
import sys
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
from dotenv import load_dotenv
load_dotenv()
# os.environ['LANGCHAIN_TRACING'] = 'true'
# os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
# os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGSMITH_API_KEY")
# os.environ['LANGCHAIN_PROJECT'] = 'GAIDO-EXP'

from functions.prereq import llm
from functions.fx import answer_question, react_agent
from typing import Literal
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from my_agent.user_state import UserProfile
from my_agent.preferences import get_preferences_questions


# ------------------------------------------------------------------------------------------------
# Creating custom preferences questions based on user profile 
# ------------------------------------------------------------

def preferences_node(state: UserProfile) -> Command[Literal["policy_match","preferences"]]:
    """Collect all User Preferrences before Recommending Policies"""
    
    # Initialize preferences structure if not already set
    if not hasattr(state, 'preferences_data') or not state.preferences_data:
        questions = get_preferences_questions(state).key_question
        state.preferences_data = [
            {'question': q, 'response': None} for q in questions
        ]
        state.current_question_index = 0


    # Check if we have completed all questions
    if state.current_question_index >= len(state.preferences_data):
        return Command(
            goto="policy_match",
            update={
                "preferences_collected": True,
                "profiling_stage": "complete",
                "interaction_count": state.interaction_count + 1
            }
        )

    # Get current question
    current_pref = state.preferences_data[state.current_question_index]
    
    # Ask the current question
    response = interrupt(current_pref['question'])
    
    # Check if user wants to skip or quit the preferences collection
    if response.lower() in ['skip', 'quit', 'stop', 'not interested', 'don\'t want to answer', 'move on', 'don\'t care', 'irrelevant', 'not applicable']:
        return Command(
            goto="policy_match",
            update={
                "preferences_collected": True,
                "profiling_stage": "skipped",
                "interaction_count": state.interaction_count + 1
            }
        )
    else:
        # Store the response
        state.preferences_data[state.current_question_index]['response'] = response
        
        # Move to next question
        return Command(
            goto="preferences",  # Loop back to same node
            update={
                "preferences_data": state.preferences_data,
                "current_question_index": state.current_question_index + 1,
                "profiling_stage": f"question_{state.current_question_index}",
                "interaction_count": state.interaction_count + 1
            }
        )


# ------------------------------------------------------------------------------------------------
# Creating policy match node 
# ------------------------------------------------------------------------------------------------

from functions.prompts import feature_guidelines
from functions.fx import get_feature_recommendation
from langchain import hub as prompts
final_recommendation_prompt = prompts.pull("final_recommendation_prompt")
from functions.fx import answer_question

def policy_match_node(state: UserProfile)-> Command[Literal["preferences","query_handling","confirmation"]]:
    
    # """Match policies to user preferences and recommend 3 most suitable options"""

    if not state.preferences_collected:
        # If preferences are not collected yet, collect them first
        return Command(
            goto="preferences",
            update={
                "interaction_count": state.interaction_count + 1
            }
        )
    
    # Use LLM to analyze user preferences and recommend policies
    User_Profile = f'''
    - Family Members: {state.family_members}
    - Age of family members: {state.age}
    - Has Pre-existing Conditions: {state.has_pre_existing_conditions}
    - Pre-existing conditions: {', '.join(state.pre_existing_conditions) if state.pre_existing_conditions else "None"}
    - User_Prefence Data: {[pref for pref in state.preferences_data if pref.get('response') is not None]}
    '''
    
    feature_recommendation = get_feature_recommendation(User_Profile)
    # Generate policy recommendations using the LLM
    reco_prompt =final_recommendation_prompt.format(user_profile=User_Profile, feature_recommendation=feature_recommendation)
    recommendations = llm.invoke(reco_prompt).content
    # recommendations = recommendations.content if hasattr(recommendations, 'content') else str(recommendations)

    opinion = interrupt(recommendations +"\n**What would you like to do next?**\nYou can:\n  - ✅ Type **'Yes'** to proceed with these recommended policies\n  - ❓ Ask any question about the policies (recommended or any other)\n  - 🔁 Type **'Update Preferences'** if you'd like to revise your requirements\nI'm here to help you with whichever step you choose!\n")
    if opinion.lower() == "yes":
        return Command(
            goto="confirmation",  # Loop back to same node
            update={
                "policy_match_done": True,
                "recommeneded_policies": recommendations,
                "interaction_count": state.interaction_count + 1
            }
        )
    
    elif opinion.lower() == "no":
        print("User requested changes, going back to preferences")
        return Command(
            goto="preferences",
            update={
                "preferences_collected": False,  # Reset this to collect preferences again
                "profiling_stage": "preferences",
                # "interaction_count": state.interaction_count + 1,
                "coverage_type": None ,
                "budget_range": None ,
                "specific_benefits": None ,
                "high_deductible_preference": None 
            }
        )
        
    else:
        # Update user profile state with recommendations and proceed to query handling
        return Command(
            goto="query_handling",
            update={
                "user_query": opinion,
                "policy_match_done": True,
                "recommeneded_policies": recommendations,
                # "interaction_count": state.interaction_count + 1
                # Removed the conflicting "profiling_stage" update
            }   
        )
    
# ------------------------------------------------------------------------------------------------
# Creating query handling node 

def query_handling_node(state: UserProfile)->Command[Literal["confirmation","query_handling"]]:
    """Handle user queries about policies using RAG_tool"""
    from langchain_core.messages import HumanMessage

    print('in query_handling_node_with_RAG')

    if state.query_handling_done == True:
        return Command(
                goto="confirmation",
                update={
                    "query_handling_done": True,
                    "user_query": None,  # Reset for potential future queries
                }
            )
    
    # If we don't have a user query yet, get one
    if not state.user_query and not state.query_handling_done:
        query_prompt = interrupt("What would you like to know about insurance policies?")
        return Command(
            goto="query_handling",  # Loop back to handle the query
            update={
                "user_query": query_prompt,
                "interaction_count": state.interaction_count + 1
            }
        )
    
    # We have a query, use RAG_tool to get the answer
    # Add validation to check if query is empty or only whitespace
    if not state.user_query or state.user_query.isspace():
        query_prompt = interrupt("Please provide a specific question about insurance policies:")
        return Command(
            goto="query_handling",
            update={
                "user_query": query_prompt,
                "interaction_count": state.interaction_count + 1
            }
        )
        
    # Invoke the RAG tool with the user's query
    
    answer = react_agent.invoke({"messages":[HumanMessage(content=state.user_query)]})
    answer = answer['messages'][-1].content
    
    # Format the response message
    response_message = f"{answer}"
    
    # Ask if user has more questions
    follow_up = interrupt(f"{response_message}\n\nDo you have any more questions? (Type your question or 'no' to finish)")
    
    if follow_up.lower() == 'no':
        # User is done asking questions
        return Command(
            goto="confirmation",
            update={
                "query_handling_done": True,
                "user_query": None,  # Reset for potential future queries
                "interaction_count": state.interaction_count + 1
            }
        )
    else:
        # User has another question
        return Command(
            goto="query_handling",  # Loop back to handle the new query
            update={
                "user_query": follow_up,
                "interaction_count": state.interaction_count + 1
            }
        )

# ------------------------------------------------------------------------------------------------
# Creating confirmation node 
# ------------------------------------------------------------------------------------------------

def confirmation_node(state: UserProfile)->Command[Literal["__end__","preferences"]]:
    """Confirm collected information and allow corrections."""
    # Generate a comprehensive summary of collected information
    print('in confirmation_node')
    
    # Create a summary of user preferences and recommendations
    summary_content = f"""
    Here's a summary of your insurance preferences:
    
    Coverage type: {state.coverage_type}
    Budget range: {state.budget_range}
    Specific benefits desired: {state.specific_benefits}
    High deductible preference: {state.high_deductible_preference}
    
    Based on these preferences, we've recommended the following policies:
    {state.recommeneded_policies}
    """
    
    # Ask for confirmation
    confirmation_message = f"{summary_content}\n\nDoes this summary look correct? Would you like to proceed with these recommendations or make any changes to your preferences? (Type 'proceed' to continue or specify what you'd like to change)"
    print("Sending confirmation message to user")
    confirmation_response = interrupt(confirmation_message)
    
    # Handle user's response
    if confirmation_response.lower() == 'proceed' or 'yes' in confirmation_response.lower() or 'correct' in confirmation_response.lower():
        # User confirms, proceed to completion
        print("User confirmed, proceeding to END")
        return Command(
            goto=END,
            update={
                "recommendation_confirmation_done": True,
                "profiling_stage": "complete",
                "interaction_count": state.interaction_count + 1
            }
        )
    else:
        # User wants to make changes, go back to preferences
        print("User requested changes, going back to preferences")
        return Command(
            goto="preferences",
            update={
                "preferences_collected": False,  # Reset this to collect preferences again
                "profiling_stage": "preferences",
                "interaction_count": state.interaction_count + 1,
                # Optionally reset specific fields based on user feedback
                "coverage_type": None if "coverage" in confirmation_response.lower() else state.coverage_type,
                "budget_range": None if "budget" in confirmation_response.lower() else state.budget_range,
                "specific_benefits": None if "benefit" in confirmation_response.lower() else state.specific_benefits,
                "high_deductible_preference": None if "deductible" in confirmation_response.lower() else state.high_deductible_preference
            }
        )

# ------------------------------------------------------------------------------------------------
# Creating the workflow graph
# ------------------------------------------------------------------------------------------------

recommendation_workflow = StateGraph(UserProfile)

# Add all nodes
recommendation_workflow.add_node("preferences", preferences_node)
recommendation_workflow.add_node("policy_match", policy_match_node)
recommendation_workflow.add_node("query_handling", query_handling_node)
recommendation_workflow.add_node("confirmation", confirmation_node)

# Set entry point
recommendation_workflow.set_entry_point("preferences")
recommendation_workflow.add_edge(START, "preferences")
recommendation_workflow.set_finish_point("confirmation")
recommendation_graph = recommendation_workflow.compile()


# ------------------------------------------------------------------------------------------------
# Running the workflow graph
# ------------------------------------------------------------------------------------------------  


# recommendation_graph = recommendation_workflow.compile(checkpointer=MemorySaver())

# initial_state = UserProfile(
#     new_user=True,
#     greeting_done=False,
#     personal_info_collected=False,
#     health_info_collected=False,
#     confirmation_done=False
# )

# config = {"configurable": {"thread_id": "unique_conversation_id"}}

# recommendation_graph.invoke(initial_state, config=config)
# state = recommendation_graph.get_state(config=config)
# print(state.values)
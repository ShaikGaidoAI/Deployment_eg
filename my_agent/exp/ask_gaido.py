import os
import sys
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from functions.prereq import llm
from functions.prompts import initial_recommendation_guidelines
from my_agent.user_state import UserProfile
from my_agent.profile_update import update_profile
from my_agent.follow_up import follow_up_userprofile
from typing import List


from pydantic import BaseModel, Field
import asyncio
from typing import Optional



# ------------------------------------------------------------------------------------------------
# Query classification
# ------------------------------------------------------------------------------------------------

class QueryResponse(BaseModel):
    # Purpose: This class defines the structure for classifying user queries about insurance
    # Why: To standardize query classification and enable appropriate response handling
    # How: Uses Pydantic BaseModel to validate query types and ensure consistent categorization
    # The query_type field will be populated by the LLM to classify incoming user queries
    # into predefined categories for routing to the appropriate response handler
    query_type: str = Field(
        description=(
            "Classifies the query into one of the following categories:\n"
            "- policy_recommendation_request: If the user is asking for policy recommendations\n" 
            "- service_information_request: If the user is asking for information about services like claims, renewals, etc.\n"
            "- policy_information_request: If the user is asking for details about specific policiy details\n"
            "- insurer_information_request: If the user is asking for information about insurance providers like HDFC ERGO, ICICI Lombard, etc."
            "- analysis_request: If the user is asking for deep analysis on any health insruance related topic, be it recommendations, comparison, validation etc.\n"
            "- other: If the user's query does not fit into the above categories"
        )
    )


# ------------------------------------------------------------------------------------------------
# Main function for handling any query coming to gaido
# ------------------------------------------------------------------------------------------------

def ask_gaido(query, state):
    
    # Clasify the query into one of the following categories
    structured_llm = llm.with_structured_output(QueryResponse)
    response = structured_llm.invoke(query)
    
    # If the query is a policy recommendation request, generate the initial recommendation
    if response.query_type == "policy_recommendation_request":
      
        if state.has_missing_profile_info():
            # print("User asked a query about policy recommendations")
            # Generate initial recommendation and ask follow up questions for missing information
            # ------------------------------------------------------------------------------------------------
            print("state before update")
            print("--------------------------------")
            print(state.get_summary())
            profile_update = update_profile(query, state)
            state.update_state(name=profile_update.name, 
                               age=profile_update.age, 
                               family_members=profile_update.family_members, 
                               has_pre_existing_conditions=profile_update.has_pre_existing_conditions, 
                               pre_existing_conditions=profile_update.pre_existing_conditions)
            print("state after update")
            print("--------------------------------")
            print(state.get_summary())
            # ------------------------------------------------------------------------------------------------
            # Generate initial recommendation
            initial_reco_answer = initial_recommendation(query, state)
            # Generate follow up question
            next_question = follow_up_userprofile(state)
            # Print initial recommendation and follow up question
            print(f"{initial_reco_answer.content}\n\n{next_question}")
            # Get user response and update profile
            user_updates = input()
            profile_update = update_profile(f"""question: {next_question} + response: {user_updates}""", state)
            state.update_state(name=profile_update.name,    
                               age=profile_update.age, 
                               family_members=profile_update.family_members, 
                               has_pre_existing_conditions=profile_update.has_pre_existing_conditions, 
                               pre_existing_conditions=profile_update.pre_existing_conditions)
            print("state after update")
            print("--------------------------------")
            print(state.get_summary())

            while state.has_missing_profile_info():
                next_question = follow_up_userprofile(state)
                user_updates = input(f"{next_question}")
                profile_update = update_profile(f"""question: {next_question} + response: {user_updates}""", state)
                state.update_state(name=profile_update.name, 
                                   age=profile_update.age, 
                                   family_members=profile_update.family_members, 
                                   has_pre_existing_conditions=profile_update.has_pre_existing_conditions, 
                                   pre_existing_conditions=profile_update.pre_existing_conditions)
                print("state after update")
                print("--------------------------------")
                print(state.get_summary())
            
        





    # If the query is a insurer information request, provide details about the insurer
    elif response.query_type == "insurer_information_request":
        print("User asked a query about insurer information")
        return "PLEASE WAIT, I AM WORKING ON IT"

    # If the query is a general information request, provide a general explanation
    elif response.query_type == "service_information_request":
        print("User asked a query about service information")
        return "PLEASE WAIT, I AM WORKING ON IT"
    
    # If the query is a policy information request, provide details about the policy
    elif response.query_type == "policy_information_request":
        print("User asked a query about policy information")
        return "PLEASE WAIT, I AM WORKING ON IT"



# ------------------------------------------------------------------------------------------------
# Initial recommendation - Based on initial query or partial user profile
# ------------------------------------------------------------------------------------------------

def initial_recommendation(query, state):
    prompt = f"""You are an expert insurance advisor providing initial policy recommendations based on whichever information is available else you can ask follow up questions.
    
    TASK:
    - Analyze the user's query {query} and profile information {state}
    - Provide 2-3 relevant policy names most suited to the user's profile {state} based on {initial_recommendation_guidelines}
    - Provide a 1 line explanation of why it may be suitable (prioritize here. Don't show too much information)
    
    
    <OUTPUT FORMAT> - This is the format you must follow. Don't hallucinate.
    ------------------------------------------------------------------------------------------------
    GENERATE THE RESPONSE IN THE FOLLOWING FORMAT AND NOTHING ELSE:

    Hey, {state.name if state.name else "there"} 🚀 

    Here are some preliminary policy recommendations crafted just for you:

    1. <Policy Name 1> - <1 line explanation of why it may be suitable>
    2. <Policy Name 2> - <1 line explanation of why it may be suitable>
    3. <Policy Name 3> - <1 line explanation of why it may be suitable>
    
    Add a friendly conversational tone to the response.

    <OUTPUT FORMAT>
    ------------------------------------------------------------------------------------------------
    
    Example 1:
    <Policy Name 1> - <1 Given your <some facts from query> and <some facts from profile>, this policy may be suitable for you>
    <Policy Name 2> - <1 Given your <some facts from query> and <some facts from profile>, this policy may be suitable for you>
    <Policy Name 3> - <1 Given your <some facts from query> and <some facts from profile>, this policy may be suitable for you>
    
    """
    
    return llm.invoke(prompt)



query = "any good plans for maternity?"
state = UserProfile()
response = ask_gaido(query, state)
print(response)

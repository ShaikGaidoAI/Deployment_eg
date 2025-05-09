from typing import Literal

from langgraph.graph import END
from langgraph.types import Command, interrupt

from functions.prereq import llm, QueryResponse
from functions.fx import initial_recommendation
from my_agent.user_state import UserProfile
from my_agent.profile_update import update_profile
structured_llm = llm.with_structured_output(QueryResponse)

# user_1 = UserProfile(
#     name="Sudhakar",
#     age=[30, 28, 5],  # Ages for self, spouse, and child respectively
#     contact_info="john.doe@example.com",
#     family_members=["self", "spouse", "child"],
#     has_pre_existing_conditions=True,
#     pre_existing_conditions=["Hypertension"],
# )

# import json, os
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

# def get_state(id: str) -> Optional[UserProfile]:
#     response = supabase.table("user_profiles").select("state").eq("id", id).execute()
    
#     if not response.data:
#         return None  # or raise an exception/log it
    
#     state_data = response.data[0]["state"]
#     return UserProfile(**state_data)


# id = str(uuid.uuid4())  # Generate a valid UUID string

# ============================================================================
# Supervisor Node
# ============================================================================
# This module contains the supervisor node that coordinates transitions between
# different agent workflows. It handles routing user queries to the appropriate
# agent based on the current state and user intent.
#
# The supervisor manages:
# - Onboarding workflow
# - Recommendation workflow
# - Policy information requests
# - Transitions between different states
# ============================================================================

def supervisor_node(state: UserProfile)->Command[Literal["onboarding_agent", "recommendation_agent", "__end__", "ask_gaido", "policy_info", "policy_comparison"]]:
    """Supervisor node that coordinates transitions between onboarding and recommendation agents"""
    
    # ------------------------------------------------------------------------------------------------
    # State Initialization
    # ------------------------------------------------------------------------------------------------
    
    # # Check if this is the first interaction (state hasn't been initialized yet)
    # if state.interaction_count == 0:
    #     try:
            
    #         # state = user_1
    #         # Initialize the state with data from the database
    #         state = get_state(id)

    #         # Update the state with data from the database
    #         return Command(
    #             goto="ask_gaido",
    #             update={
    #                 **state.model_dump(),
    #                 "interaction_count": 1
    #             }
    #         )
    #     except Exception as e:
    #         # Log the error but continue with default state if database fetch fails
    #         print(f"Error initializing from database: {e}")
    #         # Continue with normal flow

    if state.greeting_done == False:
        response = f'''Welcome {state.name if state.name else "!"} 😊
            I'm here to make your health insurance journey simple and stress-free.
            From finding the right coverage to understanding your options, 
            I'll help you choose the best health policy for your needs. Let's get started — your health is in good hands!'''
        
        state.user_intent_query = interrupt(response)
        
        
        # UPDATING CHAT HISTORY
        state.messages.append("assistant: " + response + "user: " + state.user_intent_query)
        # update_state(state, id)
        return Command(
            goto="ask_gaido",
            update={
                **state.model_dump(),  # Start with current state
                "greeting_done": True,
            }
        )
    
    
    if state.user_intent_query == None:
        state.user_intent_query = interrupt("How can I help you today?")
        state.messages.append("assistant: How can I help you today? " + "user: " + state.user_intent_query)
    response = structured_llm.invoke(state.user_intent_query)

    
    # ------------------------------------------------------------------------------------------------
    # Handle policy recommendation intent
    # This section processes user requests for policy recommendations:
    # 1. Updates user profile in background if information is missing
    # 2. Provides initial recommendations based on available information
    # 3. Offers to collect more details through onboarding workflow
    # 4. Routes to appropriate next steps based on user response
    # ------------------------------------------------------------------------------------------------
    if response.query_type == "policy_recommendation_request":
      
        if state.has_missing_profile_info():
            
            
            # Schedule profile update as a background task and continue without waiting
            import threading
            
            def background_profile_update(query, user_state):
                profile_update = update_profile(query, user_state)
                user_state.update_state(name=profile_update.name, 
                                      age=profile_update.age, 
                                      family_members=profile_update.family_members, 
                                      has_pre_existing_conditions=profile_update.has_pre_existing_conditions, 
                                      pre_existing_conditions=profile_update.pre_existing_conditions)
            
            # Start the profile update in a background thread
            threading.Thread(target=background_profile_update, 
                           args=(state.user_intent_query, state), 
                           daemon=True).start()
            
            
            initial_reco_answer = initial_recommendation(state.user_intent_query, state)
            follow_up_text = """These are just preliminary recommendations! To help me refine them and find the perfect plan for you, I'd love to learn a bit more about you."""
            proceed_question = "Would you like to proceed with more details. Type YES or NO "
            
            user_output = f"{initial_reco_answer}\n\n{follow_up_text}\n\n{proceed_question}"
            user_updates = interrupt(user_output)
            # UPDATING CHAT HISTORY
            state.messages.append("assistant: " + user_output + "user: " + user_updates)
            
            
            if user_updates.lower() == "yes":
                # update_state(state, id)
                return Command(
            goto="onboarding_agent",
            update={
                **state.model_dump(),
                "user_query": state.user_intent_query,
                "greeting_done": state.greeting_done,
                "recommeneded_policies": initial_reco_answer,
                # **state.model_dump(),  # Start with current state
                "current_workflow": "onboarding",
                # Additional context that might be useful for the onboarding agent
                "interaction_count": state.interaction_count + 1,
            }
            )

            else:
                # update_state(state, id)
                return Command(
                    goto="ask_gaido",
                    update={
                        **state.model_dump(),
                        "interaction_count": state.interaction_count + 1,
                    }
                )

    # ------------------------------------------------------------------------------------------------
    # Handle policy comparison intent
    # This section processes user requests for policy comparisons:
    # 1. Routes to policy comparison node
    # 2. Updates state with current workflow and user query
    # 3. Increments interaction count
    # ------------------------------------------------------------------------------------------------
    if response.query_type == "policy_comparison_request":
        # update_state(state, id)
        return Command(
            goto="policy_comparison",
            update={
                **state.model_dump(),
                "current_workflow": "policy_comparison",
                "user_query": state.user_intent_query,
                "interaction_count": state.interaction_count + 1,
            }
        )
     
    # ------------------------------------------------------------------------------------------------
    # Handle policy information request
    # This section processes user requests for policy information:
    # 1. Routes to policy information node
    # 2. Updates state with current workflow and user query
    # 3. Increments interaction count
    # ------------------------------------------------------------------------------------------------
    elif response.query_type == "policy_information_request" or response.query_type == "insurer_information_request" or response.query_type == "service_information_request":
        # update_state(state, id)
        return Command(
            goto="policy_info",
            update={
                **state.model_dump(),
                "current_workflow": "policy_info",
                "user_query": state.user_intent_query,
                "interaction_count": state.interaction_count + 1,
            }
        )
    
    # ------------------------------------------------------------------------------------------------
    # Handle policy analysis request
    # This section processes user requests for policy analysis:
    # 1. Routes to policy comparison node
    # 2. Updates state with current workflow and user query
    # 3. Increments interaction count
    # ------------------------------------------------------------------------------------------------
    elif response.query_type == "analysis_request":
        # update_state(state, id)
        return Command(
            goto="policy_comparison",
            update={
                **state.model_dump(),
                "current_workflow": "policy_comparison",
                "user_query": state.user_intent_query,
                "interaction_count": state.interaction_count + 1,
            }
        )

    # ------------------------------------------------------------------------------------------------
    # Handle onboarding workflow
    # This section processes user requests for onboarding:
    # 1. Routes to onboarding agent if onboarding is not complete
    # 2. Updates state with current workflow and interaction count
    # ------------------------------------------------------------------------------------------------
    if not state.onboarding_complete:
        # update_state(state, id)
        return Command(
            goto="onboarding_agent",
            update={
                **state.model_dump(),
                "current_workflow": "onboarding",
                "interaction_count": state.interaction_count + 1,
            }
        )
    
    # If onboarding is complete but recommendation is not, go to recommendation agent
    if state.onboarding_complete and not state.recommendation_complete:
        # update_state(state, id)
        return Command(
            goto="recommendation_agent",
            update={
                **state.model_dump(),
                "current_workflow": "recommendation",
                "interaction_count": state.interaction_count + 1,
            }
        )
    
    # If both are complete, end the process
    if state.onboarding_complete and state.recommendation_complete:
        # update_state(state, id)
        return Command(
            goto=END,
            update={
                **state.model_dump(),
                "current_workflow": "complete",
                "interaction_count": state.interaction_count + 1,
            }
        )
    
    # Default case - stay in supervisor
    # update_state(state, id)
    return Command(
        goto="ask_gaido",
        update={
            **state.model_dump(),
            "current_workflow": "supervisor",
            "interaction_count": state.interaction_count + 1,
        }
    ) 
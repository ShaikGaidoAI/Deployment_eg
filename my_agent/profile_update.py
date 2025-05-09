import os
import sys
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from pydantic import BaseModel, Field
from typing import Optional, List
from .user_state import UserProfile
import asyncio

from functions.prereq import llm, llm_openai, llm_gemini, llm_anthropic

# ------------------------------------------------------------------------------------------------
# Update the user profile based on the query
# ------------------------------------------------------------------------------------------------

class ProfileUpdate(BaseModel):
    
    name: Optional[str] = Field(
        description=(
            "Name of the user"
        ),
        default=None
    )

    family_members: Optional[List[str]] = Field(
        description=(
            "List of family members including the user, title him as self"
        ),
        default_factory=list
    )
    
    age: Optional[List[int]] = Field(
        description=(
            "List of ages of family members"
        ),
        default_factory=list
    )

    has_pre_existing_conditions: Optional[bool] = Field(
        description=(
            "Whether the user has any pre-existing conditions"
        ),
        default=None
    )
    
    pre_existing_conditions: Optional[List[str]] = Field(
        description=(
            "List of pre-existing conditions of family members"
        ),
        default_factory=list
    )


def update_profile(query: str, state: UserProfile) -> ProfileUpdate:
    """
    Extract relevant user profile updates based on information presented in the query.
    Don't hallucinate. Only extract information that is explicitly mentioned in the query, which you are sure about.
    

    Args:
        query: The user's query string
        state: Current user profile state
        
    Returns:
        ProfileUpdate object with extracted information relevant to the query
    
    """
    profile_update_prompt = f"""
    IMPORTANT: Only extract information that is EXPLICITLY mentioned in the query. Do not make assumptions or infer information.
    If information is not directly stated, leave those fields as None or empty lists.
    
    Query: {query}

    Task: Extract ONLY the following information if explicitly stated:
    1. Name (exact name mentioned)
    2. Family members (exact relationships/names mentioned)
    3. Ages (exact numbers mentioned)
    4. Pre-existing conditions (exact conditions mentioned)
    5. Has pre-existing conditions (only if explicitly stated yes/no)

    Current profile state:
    {state.model_dump_json(indent=2)}

    Rules:
    - Do not infer or assume any information
    - Only include information directly stated in the query
    - Use None for missing information
    - Use empty lists for missing lists
    - Maintain consistency with existing profile data
    - If query is just asking for information without providing any, return empty/None values

    Example outputs:
    "I am John, 35 years old" -> 
    {{
        "name": "John",
        "age": [35],
        "family_members": ["self"],
        "has_pre_existing_conditions": null,
        "pre_existing_conditions": []
    }}

    "Looking for insurance" ->
    {{
        "name": null,
        "age": [],
        "family_members": [],
        "has_pre_existing_conditions": null,
        "pre_existing_conditions": []
    }}
    """
    
    # structured_llm = llm.with_structured_output(ProfileUpdate)
    # response = structured_llm.invoke(profile_update_prompt)

    # structured_llm_anthropic = llm_anthropic.with_structured_output(ProfileUpdate)
    # response_anthropic = structured_llm_anthropic.invoke(profile_update_prompt)

    structured_llm_openai = llm_openai.with_structured_output(ProfileUpdate)
    response_openai = structured_llm_openai.invoke(profile_update_prompt)

    # # Step 4: Get response from Gemini
    # triangulation_prompt = f"""
    # You are an expert in triangulating the accurate state updates from the user query.
   

    # Given responses from 3 different LLMs, extract the most accurate state updates.

    # User query: {query}
    # Current state: {state.get_summary()}
    # Response from LLM 1: {response}
    # Response from LLM 2: {response_anthropic}
    # Response from LLM 3: {response_openai}
    # """

    # structured_llm = llm.with_structured_output(ProfileUpdate)
    # response = structured_llm.invoke(triangulation_prompt)

    return response_openai


# ------------------------------------------------------------------------------------------------
# Local testing of the implementation so far
# ------------------------------------------------------------------------------------------------

# # Test case 1: Basic parent query
# user_profile = UserProfile()
# query = "Give me a good health insurance policy for parents"
# update = update_profile(query, user_profile)
# print("Test 1:", update)

# # Test case 2: Query with specific names and ages
# user_profile = UserProfile()
# query = "I need insurance for my father John who is 65 and mother Mary who is 60"
# update = update_profile(query, user_profile)
# print("Test 2:", update)

# # Test case 3: Query with pre-existing conditions
# user_profile = UserProfile()
# query = "My dad has diabetes and high blood pressure, need insurance coverage"
# update = update_profile(query, user_profile)
# print("Test 3:", update)

# # Test case 4: Query with self information
# user_profile = UserProfile()
# query = "I am Sarah, 35 years old looking for family coverage"
# update = update_profile(query, user_profile)
# print("Test 4:", update)

# # Test case 5: Complex family query
# user_profile = UserProfile()
# query = "Need insurance for my family - myself (Tom, 40), wife Jane (38), and kids Mike (10) and Lisa (8)"
# update = update_profile(query, user_profile)
# print("Test 5:", update)

# # Test case 6: Senior citizens with conditions
# user_profile = UserProfile()
# query = "My parents are 70 and 68, mom has arthritis and dad had a heart attack last year"
# update = update_profile(query, user_profile)
# print("Test 6:", update)

# # Test case 7: Single parent query
# user_profile = UserProfile()
# query = "Looking for insurance for my mother who is 55 years old"
# update = update_profile(query, user_profile)
# print("Test 7:", update)

# # Test case 8: Multiple conditions query
# user_profile = UserProfile()
# query = "My father aged 62 has diabetes, hypertension and had bypass surgery"
# update = update_profile(query, user_profile)
# print("Test 8:", update)

# # Test case 9: Vague query
# user_profile = UserProfile()
# query = "What are the best insurance policies available?"
# update = update_profile(query, user_profile)
# print("Test 9:", update)

# # Test case 10: Query with partial information
# user_profile = UserProfile()
# query = "Need health insurance for parents, father has diabetes"
# update = update_profile(query, user_profile)
# print("Test 10:", update)
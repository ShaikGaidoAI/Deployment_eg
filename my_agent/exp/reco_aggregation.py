import os
import sys
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from functions.prereq import llm, llm_openai, llm_gemini, llm_anthropic
from functions.fx import get_feature_recommendation
from functions.prompts import final_recommendation_prompt
from my_agent.user_state import UserProfile

user_initial_profile = UserProfile(
    name="Sudhakar",
    family_members=["self", "wife"],
    age=[50, 45],
    pre_existing_conditions=["Yes"],
    preferences_data=[{"diabetes": "yes", "hypertension": "yes", "affordable": "yes"}],
    messages=[]
)


feature_recommendation = get_feature_recommendation(user_initial_profile)
reco_prompt = final_recommendation_prompt.format(user_profile=user_initial_profile, feature_recommendation=feature_recommendation)

import asyncio

# ------------------------------------------------------------------------------------------------
# Getting recommendation responses from multiple modelss
# ------------------------------------------------------------------------------------------------

def get_all_llm_responses(prompt):
    async def get_llm_responses():
        tasks = [
            llm_openai.ainvoke(prompt),
            llm_gemini.ainvoke(prompt),
            llm_anthropic.ainvoke(prompt)
        ]
        reco_response_openai, reco_response_gemini, reco_response_anthropic = await asyncio.gather(*tasks)
        return reco_response_openai, reco_response_gemini, reco_response_anthropic

    return asyncio.run(get_llm_responses())

reco_response_openai, reco_response_gemini, reco_response_anthropic = get_all_llm_responses(reco_prompt)


# ------------------------------------------------------------------------------------------------
# Aggregation of recommendations
# ------------------------------------------------------------------------------------------------

def aggregate_recommendations(openai_response, gemini_response, anthropic_response):
    """
    Aggregates recommendations from multiple LLMs and generates a final triangulated response.
    """
    aggregation_prompt = f"""
    You are a recommendation aggregation agent. Your task is to analyze recommendations from multiple AI models 
    and generate a comprehensive, triangulated response. Consider the following recommendations:

    OpenAI's Recommendation:
    {openai_response}

    Gemini's Recommendation:
    {gemini_response}

    Anthropic's Recommendation:
    {anthropic_response}

    Please analyze these recommendations and:
    1. Identify common themes and suggestions across all models
    2. Note any unique valuable insights from individual models
    3. Resolve any contradictions between recommendations
    4. Create a synthesized, coherent recommendation that leverages the best insights from all models
    5. Ensure the final recommendation is clear, actionable, and well-reasoned

    Provide your final triangulated recommendation.
    """

    # Using OpenAI as the reflection model for triangulation
    final_triangulated_response = llm_openai.invoke(aggregation_prompt)
    return final_triangulated_response

# ------------------------------------------------------------------------------------------------
# Testing the final recommendation locally
# ------------------------------------------------------------------------------------------------

# final_recommendation = aggregate_recommendations(
#     reco_response_openai,
#     reco_response_gemini,
#     reco_response_anthropic
# )

# print("\nFinal Triangulated Recommendation:")
# print(final_recommendation.content)





































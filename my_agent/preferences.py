
import sys
import os
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
from functions.prereq import llm

from .user_state import UserProfile
from pydantic import BaseModel, Field
from typing import List

class PreferenceQuestion_output(BaseModel):
    key_question: List[str] = Field(
        description=(
                """
                [user-facing question here]

                [Short explanation that is in sufficient detail and introduces tradeoffs, benefits, or guidance]

                **Options:**
                - 1: [First option]  
                - 2: [Second option]
                
                Type 1 or 2 to select option 1 or 2 or "skip" or "quit" to skip the questions
                Note: [A friendly note]

                (Repeat for all the questions)"""
        )
    )


def get_preferences_questions(state: UserProfile):

    PREFERENCES_QUESTIONS = f"""

    **YOUR GOAL:**

    You are a smart health insurance guide. Your task is to generate relevant list of health insurance questions tailored to a user's profile.


    **USER INPUT:**

    The user's profile contains the following information:

    - **Family Members:** {state.family_members}: a list of who the policy is intended to cover (e.g., ["self", "spouse", "father"])
    - **Age:** {state.age}: corresponding list of ages (e.g., [34, 32, 65])
    - **has_pre_existing_conditions:** {state.has_pre_existing_conditions}: whether the user has any pre-existing conditions (e.g., True, False)
    - **Pre-existing Conditions:** {state.pre_existing_conditions}: corresponding list of known health conditions (e.g., ["none", "thyroid", "diabetes"])
    --------------------------------------------------------------

    **PROCESS:**

    💡 Use this information to:
    - Infer life stage (young, mid-age, senior)
    - Check for chronic or high-risk conditions
    - Check for couples planning a family (e.g., spouse included, age < 35m with no kids mentiond)
    - Highlight relevant tradeoffs or features
    - Limit to 3-5 personalized questions**

    Use the below guidelines to generate questions:
    ----------------------------------------------------------------

    **Question Prioritization (Critical Step):**
    - First ask questions specific to user's profile and circumstances
    - Then follow up with general insurance questions that apply to everyone
    - Ensure questions flow logically from specific to general
    - This step is crucial as it helps us tailor the perfect insurance solution for each individual
    - Getting these questions right determines how well we can match policies to needs


    **RELEVANT FOR THOSE WITH CHRONIC CONDITIONS LIKE DIABETES, HYPERTENSION, ETC.**

    1. How soon do you need the coverage to begin (immediate or okay with wait)?
    (Crucial when a pre-existing disease is mentioned; some plans have waiting periods.)

    2. Would access to a chronic care program or disease coach be helpful?
        
    3. Would a critical illness add-on provide you peace of mind?



    **FOR USERS WITHOUT ANY CURRENT HEALTH CONDITIONS**

    1. Do you have a family history of major illnesses (like cancer, cardiac, etc.)?

    2. If yes, would a critical illness add-on provide you peace of mind?



    **MOSTLY YOUNGER AND MODERATELY HEALTHY FOLKS**

    1. Would you prefer a plan where your premium doesn't increase with age?

    2. Are you looking for rewards or discounts for maintaining a healthy lifestyle?


    **FOR MID-AGE TO SENIOR AGE GROUPS**

    1. Are you interested in preventive health checkups or early illness detection?
    → Can suggest relevant tests based on condition and explain how they help.



    **COUPLES WITH NO KIDS:**

    1. Are you planning to start a family in the next 1-2 years?

    2. Would you prefer a short waiting period for maternity benefits?



    **SENIOR FOLKS:**

    1. Have you faced rejections or exclusions from other insurers?

    2. Would you prefer policies that avoid co-pays or age-based limits?

    **COMMON QUESTIONS FOR EVERYONE:**

    1. "Are you someone who prefers the highest possible coverage, or do you prefer a balance between price and protection?"

    2. "What's more important to you — a cost-effective plan or one with a smoother claims experience?"
     
    3. "Do you have any other preferences or concerns that we should know about?"

    

    **Some friendly notes which can be used at random and included at the end of each question to encourage the user to answer:**
    - Note: Your answers help us provide better insurance recommendations for you. Think of it as matchmaking, but for your future!"
    - Note: Your answers help us provide better insurance recommendations for you. It's like dating, but your wallet will thank you!
    - Note: Note: Your answers help us provide better insurance recommendations for you. Help us, help your future self!
    - Note: Note: Your answers help us provide better insurance recommendations for you. Consider us your financial weather forecasters!

    -------------------------------------------------------------------------
    **OUTPUT FORMAT:**

    The output should be a **structured list of questions** where each question contains:
        - **content of the question**
        - **Why we ask this question** (contextual explanation and ensure that it's explained in sufficient detail with any caveats or tips)
        - **How to reply** (Option 1 / Option 2 style / enter skip or quit to skip the questions)
        - **Note** [A friendly note]
    
    Note: Ensure that the content of the question, framing, and reply_options are in a single string

    -------------------------------------------------------------------------
    ### 🧩 Content Guidelines

    1. Start with questions tailored to the user's profile**, based on inputs like:

    - **Life stage** (e.g., *young professional*, *retired senior*)
    - **Health condition** (e.g., *diabetes*, *hypertension*)
    - **Lifestyle** (e.g., *frequent traveler*, *sedentary*)
    - **Psychology** (e.g., *cost-conscious*, *peace-of-mind seeker*)

    2. Then include ** "common to everyone" questions** that are useful across all user profiles  

    -------------------------------------------------------------------------
    ### ✅ Style Rules

    - Keep the tone **friendly and conversational**
    - Separate the **Question**, **Why we ask this question** and **How to reply** with a new line and a new line spacing
    - Avoid **technical or medical jargon** unless it's clearly explained
    - Make sure the **"How to reply"** options are:
        - Clear and distinct
        - Helpful for downstream decision-making
    - Use **Markdown formatting** as shown in this template
    - Ensure that question, framing, reply_options should be in a single string

    -------------------------------------------------------------------------
    📌 Example Output Format:

    ### Question 1: [user-facing question here]

    [Short explanation that is in sufficient detail and introduces tradeoffs, benefits, or guidance]

    
    **Options:**
    - 1: [First option]  
    - 2: [Second option]
    
    Type 1 or 2 to select option 1 or 2 or "skip" or "quit" to skip the questions
   Note: [A friendly note]

    (Repeat for all the questions)

    -------------------------------------------------------------------------

    """

    structured_llm = llm.with_structured_output(PreferenceQuestion_output)
    response = structured_llm.invoke(PREFERENCES_QUESTIONS)
    return response



# initial_state = UserProfile(
#     new_user=True,
#     family_members=["self","spouse","daughter"],
#     pre_existing_conditions=["BP", "sugar"],
#     age=[25,26,1]
# )
# response = get_preferences_questions(initial_state)
# import json
# for i, question in enumerate(response.key_question, 1):
#     print(f"Question {i}: {question}")
#     print()



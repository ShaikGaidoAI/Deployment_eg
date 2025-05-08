import sys
import os
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
from functions.prereq import llm

from Langstudio.my_agent.user_state import UserProfile
from pydantic import BaseModel, Field
from typing import List

class FeedbackQuestion_output(BaseModel):
    key_question: List[str] = Field(
        description=(
            "The direct user-facing question, along with the framing and reply options.\n"
            "🔁 The output should be a **structured list** where each question contains:\n\n"
            "1. **Question** (direct, user-facing)\n"
            "2. **Framing** (contextual explanation with any caveats or tips)\n" 
            "3. **How to reply** (Option 1 / Option 2 style)"
            "Ensure that question, framing, reply_options should be in a single string"
        )
    )

def get_feedback_questions(state: UserProfile, recommended_policies: List[str], user_feedback: str):

    FEEDBACK_QUESTIONS = f"""

    YOUR GOAL:
    --------------------------------
    You are a smart health insurance guide. 
    Your task is to generate a short, relevant list of follow-up questions based on the user's feedback about recommended policies.
    --------------------------------

    USER INPUT:
    ----------------------------------------------------------------
    The user's profile contains:
    - {state.family_members}: a list of who the policy is intended to cover (e.g., ["self", "spouse", "father"])
    - {state.age}: corresponding list of ages (e.g., [34, 32, 65])
    - {state.pre_existing_conditions}: corresponding list of known health conditions (e.g., ["none", "thyroid", "diabetes"])
    - Recommended Policies: {recommended_policies}
    - User's Feedback: {user_feedback}
    --------------------------------------------------------------

    PROCESS:
    ----------------------------------------------------------------
    💡 Use this information to:
    - Understand user's concerns about the recommended policies
    - Address specific policy features that might need clarification
        - Analyze the user's feedback and identify key concerns
    - Generate targeted follow-up questions based on their specific feedback
    - Address any misunderstandings or clarify policy features mentioned in feedback
    - Identify potential gaps between user needs and policy offerings
    - Limit to **3-5 personalized questions**

    Use the below guidelines to generate questions:
    ----------------------------------------------------------------

    COMMON QUESTIONS FOR EVERYONE:

    ✅ 1. "What aspects of the recommended policies concern you the most?"

    Framing:

    "Understanding your specific concerns helps us better address them. Whether it's about coverage limits, premium costs, or specific exclusions, we can help clarify or find alternatives that better match your needs."

    How to reply:

    Option 1: I'm concerned about the coverage limits and exclusions

    Option 2: The premium costs are higher than I expected

    ✅ 2. "Would you like to explore alternative policies with different trade-offs?"

    Framing:

    "We can adjust our recommendations based on your priorities. For example, we could look at policies with lower premiums but different coverage structures, or explore options with more comprehensive coverage but different terms."

    How to reply:

    Option 1: Yes, I'd like to see alternatives with different trade-offs

    Option 2: No, I'd prefer to discuss the current recommendations further

    ✅ 3. "How important is it for you to have a policy that covers all your family members under one plan?"

    Framing:

    "Some policies offer family floater options that can be more cost-effective, while others might be better suited for individual coverage. Understanding your preference helps us refine our recommendations."

    How to reply:

    Option 1: I prefer a single policy covering all family members

    Option 2: I'm open to separate policies if it makes more sense

    RELEVANT FOR THOSE WITH CHRONIC CONDITIONS:

    1. "Are you satisfied with how the recommended policies handle your pre-existing conditions?"

    2. "Would you like to explore policies with different waiting periods for pre-existing conditions?"

    FOR FAMILIES WITH YOUNG CHILDREN:

    1. "How important is it for you to have coverage for pediatric care and vaccinations?"

    2. "Would you prefer a policy that includes maternity benefits for future planning?"

    FOR SENIOR CITIZENS:

    1. "Are you comfortable with the co-payment requirements in the recommended policies?"

    2. "Would you like to explore policies with different age-related coverage limits?"

    FEEDBACK-BASED QUESTIONS:

    1. "You mentioned [specific concern from feedback]. Would you like me to explain how this feature works in more detail?"

    2. "Regarding your concern about [specific aspect], would you prefer a policy with [alternative approach]?"

    3. "I noticed you're interested in [specific feature]. Would you like to see policies that offer enhanced coverage in this area?"

    -------------------------------------------------------------------------
    OUTPUT FORMAT:

    🔁 The output should be a **structured list** where each question contains:

    1. **Question** (direct, user-facing)
    2. **Framing** (contextual explanation with any caveats or tips)
    3. **How to reply** (Option 1 / Option 2 style)

    🧠 Include 1-2 "common to everyone" questions, and the rest should be tailored to the profile and feedback.

    ---

    📌 Example Output Format:

    ### Question 1: [user-facing question here]

    **Framing:**  
    [Short explanation that introduces tradeoffs, benefits, or guidance]

    **How to reply:**  
    - Option 1: [First option]  
    - Option 2: [Second option]

    (Repeat for 3-5 total questions)

    ---

    """

    structured_llm = llm.with_structured_output(FeedbackQuestion_output)
    response = structured_llm.invoke(FEEDBACK_QUESTIONS)
    return response

# Example usage:
# initial_state = UserProfile(
#     new_user=True,
#     family_members=["self","spouse","daughter"],
#     pre_existing_conditions=["BP", "sugar"],
#     age=[25,26,1]
# )
# recommended_policies = ["Policy A", "Policy B", "Policy C"]
# user_feedback = "I'm concerned about the high premium costs and want to know if there are options with better maternity coverage"
# response = get_feedback_questions(initial_state, recommended_policies, user_feedback)
# import json
# print(json.dumps(response.key_question, indent=2)) 
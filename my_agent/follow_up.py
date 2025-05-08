from my_agent.user_state import UserProfile
from functions.prereq import llm


def follow_up_userprofile(state):
    """
    Generate a follow-up question based on missing profile information.
    
    Args:
        state: The current UserProfile state
        
    Returns:
        A follow-up question as a string
    """
    
    # Define the questions for different missing profile fields
    name_question = """
First things first - what's your name? This will help me personalize our conversation and tailor the recommendations specifically for you."""

    family_members_question = """Who would you like to include in your health insurance coverage? 🏥👨‍👩‍👧‍👦

For example:
- Just yourself
- You and your spouse
- Your parents
- Your entire family

How to enter your response:

Simply type the family members separated by commas, like:
"self, spouse, daughter"
or
"self, mother, father"
or just
"self"

Please list all family members you'd like to insure:"""
    
    # Use the family_members data if available, otherwise use a generic message
    if state.family_members:
        family_str = ', '.join(str(member) for member in state.family_members)
        age_question = f"""Please provide the age for each family member you listed ({family_str}):

How to enter:
Enter ages in the same order, separated by commas. For example:
"35, 32, 5" 

Please enter the ages now:"""
    else:
        age_question = """Please provide your age:

How to enter:
Enter your age as a number. For example: "35"

Please enter your age now:"""

    # Use the name if available, otherwise use "there"
    name_str = state.name if state.name else "there"
    has_pre_existing_conditions_question = f"""Hey {name_str}!

💡 Knowing about any pre-existing medical conditions in the family helps us recommend plans that offer the right benefits—like shorter waiting periods, better coverage, or specialized care.

For families with multiple members, it's important to choose insurance that fits everyone's health needs—whether it's for a child, parent, or senior.

👉 To make sure we get this right, could you let us know if you or any family members 
have any pre-existing medical conditions?
(Examples: diabetes, heart conditions, high blood pressure)

Just reply with yes or no."""
    
    pre_existing_conditions_question = """Could you please list the pre-existing conditions for yourself or any family members?
(Example: diabetes, high blood pressure, asthma)

👉 Just separate them with commas so we can guide you better."""

    # Check which information is missing and return the appropriate question
    if state.has_missing_profile_info():
        if state.name is None:
            return name_question
        
        if not state.family_members:
            return family_members_question
        
        if not state.age:
            return age_question
        
        if state.has_pre_existing_conditions is None:
            return has_pre_existing_conditions_question
        
        if state.has_pre_existing_conditions and not state.pre_existing_conditions:
            return pre_existing_conditions_question
            
    # If we get here, either all information is present or we've exhausted our questions
    return """Thanks for providing all the necessary information! We now have a good understanding of your profile and needs."""


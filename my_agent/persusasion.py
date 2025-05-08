import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from user_state import UserProfile
from functions.prereq import llm
from recommendation_agent import get_feature_recommendation, final_recommendation_prompt

# Initialize user profile
user1 = UserProfile(
    name="John Doe",
    family_members=["self", "spouse", "child"],
    age=[25, 28, 1],
    has_pre_existing_conditions=False,
    pre_existing_conditions=[],
    preferences_data=[]
)

# Get initial recommendations
feature_recommendation = get_feature_recommendation(user1)
reco_prompt = final_recommendation_prompt.format(user_profile=user1, feature_recommendation=feature_recommendation)
recommendations = llm.invoke(reco_prompt)
recommendations = recommendations.content if hasattr(recommendations, 'content') else str(recommendations)

# Store recommendations in user profile
user1.recommeneded_policies = recommendations
print(user1.recommeneded_policies)

# Conversation loop
def conversation_loop(user_profile: UserProfile):
    conversation_history = []
    
    # Add initial recommendations to conversation history
    conversation_history.append({
        "role": "assistant",
        "content": f"Based on your profile, here are the recommended insurance policies:\n\n{recommendations}\n\nDo you have any questions about these recommendations?"
    })
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nThank you for the conversation! Feel free to return if you have more questions.")
            break
            
        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Prepare context for LLM
        context = {
            "user_profile": user_profile.get_summary(),
            "recommendations": user_profile.recommeneded_policies,
            "conversation_history": conversation_history
        }
        
        # Create prompt for LLM
        persuasion_prompt = f"""
        You are an insurance advisor helping a customer understand their recommended policies.
        
        User Profile:
        {context['user_profile']}
        
        Recommended Policies:
        {context['recommendations']}
        
        Conversation History:
        {[f"{msg['role']}: {msg['content']}" for msg in context['conversation_history']]}
        
        Based on the above context, provide a helpful and persuasive response to the user's latest question or concern.
        Focus on addressing their specific needs and explaining how the recommended policies match their requirements.
        """
        
        # Get response from LLM
        response = llm.invoke(persuasion_prompt)
        response = response.content if hasattr(response, 'content') else str(response)
        
        # Add assistant response to history
        conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        # Print response
        print(f"\nAssistant: {response}")

# Start the conversation
print("Welcome to your personalized insurance consultation!")
conversation_loop(user1)















import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
os.environ['LANGCHAIN_TRACING'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGSMITH_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = 'GAIDO-EXP'

import sys
print(os.getenv("LANGSMITH_API_KEY"))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.fx import RAG_tool, web_search_tool
from langchain import hub as prompts

from user_state import UserProfile
from functions.prereq import llm
llm_with_tools = llm.bind_tools([RAG_tool, web_search_tool])
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(
    model=llm,
    tools=[RAG_tool, web_search_tool]
)


from recommendation_agent import get_feature_recommendation

# Define prompts directly instead of pulling from hub
final_recommendation_prompt = prompts.pull("final_recommendation_prompt")


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
recommendations = llm_with_tools.invoke(reco_prompt)
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
            
        
        # Prepare context for LLM
        context = {
            "user_profile": user_profile.get_summary(),
            "recommendations": user_profile.recommeneded_policies,
            "conversation_history": conversation_history,
            "current_question": user_input
        }
        
        # Create prompt for LLM
        persuasion_prompt = f"""
        You are an insurance advisor helping a customer understand their recommended policies.
        You can use the following tools to help you:
        - RAG_tool: To answer questions about the recommended policies
        - web_search_tool: To answer questions about the recommended policies
        
        User Profile:
        {context['user_profile']}
        
        Recommended Policies:
        {context['recommendations']}
        
        Conversation History:
        {[f"{msg['role']}: {msg['content']}" for msg in context['conversation_history']]}

        Current Question:
        {context['current_question']}
        
        Based on the above context, provide a helpful and persuasive response to the user's latest question or concern.
        Focus on addressing their specific needs and explaining how the recommended policies match their requirements.
        Make sure to use the tools to answer the user's question if needed.
        """
        
        # Get response from LLM
        response = agent.invoke({"messages": persuasion_prompt})
        # response = response.content if hasattr(response, 'content') else str(response)
        response  = response['messages'][-1].content
        
        # Add assistant response to history
        conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
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















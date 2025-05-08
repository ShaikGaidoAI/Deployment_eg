from typing import Literal
from langgraph.types import Command
from langgraph.graph import END
from my_agent.user_state import UserProfile
from functions.fx import policy_comparison
from functions.prereq import llm
from langgraph.types import interrupt
from functions.prereq import summaries_vectorstore
from functions.fx import count_policies_compared

def policy_comparison_node(state: UserProfile) -> Command[Literal["ask_gaido", "policy_comparison", "policy_info"]]:

    
    if not state.user_intent_query:
        # If no policy summaries are provided, ask for them
        response = "I need information about the policies you'd like to compare. Could you please provide details about the policies you're interested in?"
        state.user_intent_query = interrupt(response)
        return Command(
            goto="policy_comparison",
            update={
                **state.model_dump(),
                "current_workflow": "policy_comparison",
                "user_intent_query": state.user_intent_query,
                "interaction_count": state.interaction_count + 1,
            }
        )
    
    # Get policy summaries from the state
    k = count_policies_compared(state.user_intent_query)
    policy_summaries = summaries_vectorstore.similarity_search(state.user_intent_query, k=k)
    policy_summaries = "---------------------------".join([policy_summaries[i].page_content for i in range(len(policy_summaries))])


    # Perform policy comparison
    comparison_result = policy_comparison(state, policy_summaries)
    
    # Format the response with comparison and follow-up prompt
    response = f"""Here's a detailed comparison of the policies:

{comparison_result}

Would you like to:
1. Compare different policies Again?
2. Get more details about any specific aspect?
3. Go to main menu?

Please let me know how I can help further!"""
    state.user_intent_query = interrupt(response)

    if state.user_intent_query == "3":
        return Command(
            goto="ask_gaido",
            update={
                **state.model_dump(),
                "current_workflow": "policy_comparison",
                "interaction_count": state.interaction_count + 1,
            }
        )
    elif state.user_intent_query == "1":
        return Command(
            goto="policy_comparison",
            update={
                **state.model_dump(),
                "current_workflow": "policy_comparison",
                "user_intent_query": None,
            }
        )
    elif state.user_intent_query == "2":
        return Command(
            goto="policy_comparison",
            update={
                **state.model_dump(),
                "current_workflow": "policy_info",
                "user_intent_query": None,
            }
        )

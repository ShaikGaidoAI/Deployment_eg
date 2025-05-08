from typing import Literal
from langgraph.types import Command
from my_agent.user_state import UserProfile
from functions.prereq import llm
from langgraph.types import interrupt
from functions.prereq import QueryResponse
from functions.fx import RAG_tool, web_search_tool
from functions.fx import route_policy_query
from functions.fx import evaluate_rag_response

structured_llm = llm.with_structured_output(QueryResponse)


def policy_info_node(state: UserProfile) -> Command[Literal["ask_gaido", "policy_info"]]:
    """
    Node that handles policy information requests using RAG and web search.
    Uses LLM to evaluate RAG responses and fall back to web search when needed.
    """
    print("\n" + "="*50)
    print("POLICY INFO NODE: Processing policy information request")
    print("="*50 + "\n")

    # If we don't have a user query yet, get one
    if not state.user_query:
        query_prompt = interrupt("What would you like to know about insurance policies?")
        return Command(
            goto="policy_info",
            update={
                **state.model_dump(),  # Include all existing state
                "user_query": query_prompt,
                "interaction_count": state.interaction_count + 1,
                "user_intent_query": None,
            }
        )

    # Route the query to the appropriate policy
    policy_name = route_policy_query(state.user_query)
    
    # First try RAG to get information from our database
    rag_answer = RAG_tool.invoke(state.user_query)
    
    # Evaluate if RAG response is sufficient
    if evaluate_rag_response(state.user_query, rag_answer):
        answer = rag_answer
        source = "our database"
    else:
        # If RAG response is insufficient, use web search
        answer = web_search_tool.invoke(state.user_query)
        source = "web search"
    
    # Present the information to the user
    user_response = interrupt(f"""
    Here's what I found about {policy_name} (from {source}):
    
    {answer}
    
    Would you like to:
    1. Ask another question about this policy
    2. Get information about a different policy
    3. Return to the main menu
    
    Please type your choice (1, 2, or 3):
    """)
    
    # Handle user response
    if user_response.strip() == "1":
        # Ask another question about the same policy
        return Command(
            goto="policy_info",
            update={
                **state.model_dump(),
                "user_query": None,  # Reset to get new query
                "interaction_count": state.interaction_count + 1
            }
        )
    elif user_response.strip() == "2":
        # Get information about a different policy
        return Command(
            goto="policy_info",
            update={
                **state.model_dump(),
                "user_query": None,  # Reset to get new query
                "interaction_count": state.interaction_count + 1
            }
        )
    else:
        # Return to main menu
        return Command(
            goto="ask_gaido",
            update={
                **state.model_dump(),
                "user_query": None,  # Reset query
                "interaction_count": state.interaction_count + 1
            }
        ) 
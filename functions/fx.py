## All the required functions are defined in the fx.py file
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


# Prerequisites:
from langchain import hub as prompts

from functions.prereq import llm, vectorstore, embeddings
# # from functions.prompts import Query_Rephrase_prompt_template
from functions.prompts import feature_guidelines, initial_recommendation_guidelines

# def get_rephrased_query(conv: str, query: str):
#         prompt = Query_Rephrase_prompt_template.format(
#             conv_memory=conv, 
#             Question=query, 
#         )
#         response = llm.invoke(prompt)
#         return response.content



# from deepeval.models.base_model import DeepEvalBaseLLM

# class GeminiDeepEvalModel(DeepEvalBaseLLM):
#     def __init__(self, model):
#         self.model = model

#     def generate(self, prompt: str) -> str:
#         try:
#             response = self.model.invoke(prompt)
#             return str(response)
#         except Exception as e:
#             return json.dumps({"error": str(e)})

#     async def a_generate(self, prompt: str) -> str:
#         try:
#             response = await self.model.ainvoke(prompt)
#             return str(response)
#         except Exception as e:
#             return json.dumps({"error": str(e)})

#     def get_model_name(self) -> str:
#         return "gemini-2.0-flash"

#     def load_model(self):
#         pass  # No need to load model as it's already initialized


# ------------------------------------------------------------------------------------------------
# Policy Router
# ------------------------------------------------------------------------------------------------
from functions.prereq import RoutePolicy

policy_router = llm.with_structured_output(RoutePolicy)

policy_routing_prompt = prompts.pull("policy_routing_prompt")
policy_router_chain = policy_routing_prompt | policy_router

def route_policy_query(question):
    try:
        routing_result = policy_router_chain.invoke({"question": question})
        return routing_result.policy_name  # Now a list
    except Exception:
        return ["General Insurance"]



# ------------------------------------------------------------------------------------------------  
# RAG Tool
# ------------------------------------------------------------------------------------------------


from langchain_core.tools import tool
from langchain_cohere import CohereRerank
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_llm_cache
from langchain_mongodb.cache import MongoDBAtlasSemanticCache
from langchain import hub as prompts


mongodb_atlas_uri = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
COLLECTION_NAME="GAIDO_CACHE"
DATABASE_NAME="Gaido_cache_DB"

set_llm_cache(MongoDBAtlasSemanticCache(
    embedding=embeddings,
    connection_string=mongodb_atlas_uri,
    collection_name=COLLECTION_NAME,
    database_name=DATABASE_NAME,
))

cohere_api = os.environ.get("COHERE_API_KEY")
os.environ["COHERE_API_KEY"] = cohere_api

def create_rag_chain(vectorstore):
    def retrieve_with_policy_filter(question):
        policy_name = route_policy_query(question)
        
        # Retrieve documents filtered by policy name
        # Rerank using Cohere
        reranker = CohereRerank(
            model="rerank-english-v3.0",
            top_n=2  # Final refined results
        )
        # filtered_docs = []
        final_docs = []
        for policy in policy_name:
            filtered_docs = vectorstore.similarity_search(
                question, 
                k=5,  # Retrieve more initially to allow reranking
                filter={"Policy_Name": policy}
            )
            reranked_docs = reranker.compress_documents(filtered_docs, question)
            final_docs.append(reranked_docs)
        
        reranked_docs = final_docs
        
        return reranked_docs  # Return the top 4 ranked documents

    
    # Prompt Template
    prompt_template = prompts.pull("rag_prompt_template")
    
    # RAG Chain
    rag_chain = (
        RunnableParallel({
            "context": retrieve_with_policy_filter,
            "question": RunnablePassthrough(),
            "chat_history": RunnablePassthrough()
        })
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


@tool
def RAG_tool(query: str):
    """Retrieve answers from the knowledge base containing insurance policy documents. Primarily used to respond to user queries based on stored information."""
    chain = create_rag_chain(vectorstore)
    response = chain.invoke(query,{'chat_history': []})
    return response

# ------------------------------------------------------------------------------------------------
# Web search tool
# ------------------------------------------------------------------------------------------------  

from openai import OpenAI

@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for information using OpenAI's web search preview API.
    This tool is used when RAG cannot find relevant information in the local database.
    """
    client = OpenAI()
    search_behavior = (
    "I am a health insurance advisor assisting users with information about health insurance. "
    "Unless the user explicitly requests information related to countries outside India, "
    "I will focus only on Indian health insurance providers, policies, official insurer websites "
    "(such as hdfcergo.com, starhealth.in, newindia.co.in), government portals (like irdai.gov.in), "
    "and trusted Indian comparison platforms (like policybazaar.com, coverfox.com, etc.). "
    "My goal is to retrieve accurate, relevant, and up-to-date health insurance information within the Indian context.\n"
    "Be concise and to the point. Do not add any extra information. Just provide the information asked for in the query.\n"
    "Limit the response to less than 5 search results.\n"
    "Query: {query}"
)
    
    try:
        response = client.responses.create(
            model="gpt-4.1",
            tools=[{"type": "web_search_preview"}],
            input=search_behavior.format(query=query),

        )
        return response.output_text
    except Exception as e:
        return f"Error performing web search: {str(e)}"

# ------------------------------------------------------------------------------------------------
# REACT AGENT
# ------------------------------------------------------------------------------------------------
from functions.fx import RAG_tool, web_search_tool
from langgraph.prebuilt import create_react_agent
react_agent = create_react_agent(
    model=llm,
    tools=[RAG_tool, web_search_tool],
    # prompt=
)

# ------------------------------------------------------------------------------------------------
# Feature recommendation - Based on user profile
# ------------------------------------------------------------------------------------------------

from functions.prompts import feature_guidelines

feature_recommendation_prompt = prompts.pull("feature_recommendation_prompt")

def get_feature_recommendation(user_profile):
    chain = feature_recommendation_prompt | llm
    # Run the chain
    response = chain.invoke({
        "feature_guidelines": feature_guidelines,
        "user_profile": user_profile
    })

    return response.content
# ------------------------------------------------------------------------------------------------
# Initial recommendation - Based on initial query or partial user profile
# ------------------------------------------------------------------------------------------------

def initial_recommendation(query, state):
    initial_recommendation_prompt = prompts.pull("initial_recommendation_prompt")
    chain = initial_recommendation_prompt | llm
    # Run the chain
    response = chain.invoke({
        "initial_recommendation_guidelines": initial_recommendation_guidelines,
        "query": query,
        "state": state.get_summary(),
        "name": state.name if state.name  else "there"
    })
    
    return response.content

# import asyncio
# from functions.prereq import llm_openai, llm_anthropic

# initial_recommendation_prompt = prompts.pull("initial_recommendation_prompt")

# async def initial_recommendation(query, state):
#     """
#     Get initial recommendations from multiple LLMs concurrently using asyncio
#     """
#     # Create chains for each LLM
#     # chain_gemini = initial_recommendation_prompt | llm
#     chain_openai = initial_recommendation_prompt | llm_openai
#     chain_anthropic = initial_recommendation_prompt | llm_anthropic
    
#     # Prepare the input for all chains
#     input_data = {
#         "initial_recommendation_guidelines": initial_recommendation_guidelines,
#         "query": query,
#         "state": state.get_summary()
#     }
    
#     # Run all chains concurrently
#     tasks = [
#         # chain_gemini.ainvoke(input_data),
#         chain_openai.ainvoke(input_data),
#         chain_anthropic.ainvoke(input_data)
#     ]
    
#     # Await all results
#     results = await asyncio.gather(*tasks)
    
#     # Extract content from each result
#     # gemini_response = results[0].content
#     openai_response = results[0].content
#     anthropic_response = results[1].content
    
#     # Return a dictionary with all responses
#     return {
#         # "gemini": gemini_response,
#         "openai": openai_response,
#         "anthropic": anthropic_response
#     }

# async def voting_recommendation(query, state, model_responses):
#     """
#     Use LLM to analyze responses from multiple models and generate a final recommendation
#     """
#     from langchain_core.prompts import ChatPromptTemplate
    
#     voting_prompt = ChatPromptTemplate.from_template("""
#     You are an expert insurance advisor. You've received recommendations from three different AI models 
#     for the following user query:

#     USER QUERY: {query}

#     USER PROFILE:
#     {state}
    
#     MODEL RECOMMENDATIONS:
    
#     1. OpenAI's recommendation:
#     {openai_response}
    
#     2. Anthropic's recommendation:
#     {anthropic_response}
    
#     Analyze these recommendations and provide a final recommendation that:
#     1. Considers the strengths of each model's suggestions
#     2. Integrates the most relevant policies based on the user's specific needs
#     3. Formats the response in a similar structure to the original recommendations
#     4. Explains briefly why these are the best options for the user
    
#     YOUR FINAL RECOMMENDATION:
#     """)
    
#     chain = voting_prompt | llm_openai
    
#     response = await chain.ainvoke({
#         "query": query,
#         "state": str(state.get_summary()),
#         # "gemini_response": model_responses["gemini"],
#         "openai_response": model_responses["openai"],
#         "anthropic_response": model_responses["anthropic"]
#     })
    
#     return response.content

# async def initial_reco(query, state):
#     return await initial_recommendation(query, state)
# # Get the final consolidated recommendation
# async def get_final_recommendation(query, state, results):
#     return await voting_recommendation(query, state, results)

# # final_recommendation = await voting_recommendation(query, user_profile, results)
# # print("=== Final Recommendation Based on All Models ===")
# # print(final_recommendation)

# ------------------------------------------------------------------------------------------------
# RAG Evaluation
# ------------------------------------------------------------------------------------------------

rag_evaluation_prompt = prompts.pull("rag_evaluation_prompt")
def evaluate_rag_response(query: str, rag_response: str) -> bool:
    """
    Use LLM to evaluate if the RAG response is sufficient or if we need web search.
    Returns True if RAG response is sufficient, False if we need web search.
    """
    evaluation_prompt = rag_evaluation_prompt.format(query=query, rag_response=rag_response)
    evaluation = llm.invoke(evaluation_prompt).content.strip()
    return evaluation == "SUFFICIENT"


# ------------------------------------------------------------------------------------------------
# Policy Comparison
# ------------------------------------------------------------------------------------------------
from my_agent.user_state import UserProfile
comparision_template = prompts.pull("policy_comparison_template")
def policy_comparison(state: UserProfile, policy_summaries: str):
    # Format the template and store the result
    formatted_prompt = comparision_template.format_messages(
        state=state, 
        policy_summaries=policy_summaries
    )
    # Pass the formatted messages to the LLM
    response = llm.invoke(formatted_prompt)
    return response.content

from langchain.prompts import ChatPromptTemplate
def count_policies_compared(text: str) -> int:
    """
    Counts the number of distinct insurance policies being compared in the given text.
    Returns an integer representing the count.
    """
    comparison_count_prompt = """
    You are an AI assistant that analyzes insurance-related text.

    Your task is to determine **how many different insurance policies are being compared** in the text below.

    ### Text:
    {text}

    ### Instructions:
    - Only count distinct policies that are explicitly mentioned or clearly compared.
    - Do NOT include general mentions or vague references.
    - Return ONLY the number (as an integer). Do not include any explanation or extra text.

    ### Output:
    """
    
    prompt = ChatPromptTemplate.from_template(comparison_count_prompt)
    result = llm.invoke(prompt.format_prompt(text=text).to_string()).content
    return int(result)


# ------------------------------------------------------------------------------------------------
# Agent Persuasion
# ------------------------------------------------------------------------------------------------
from langgraph.types import interrupt, Command
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(
    model=llm,
    tools=[RAG_tool, web_search_tool]
)
llm_with_tools = llm.bind_tools([RAG_tool, web_search_tool])

# # Create prompt for LLM
persuasion_prompt = """
        
# Agent Persuasion System

## User Information
**User Profile:**
{user_profile}

## Conversation Context
**Conversation History:**
{conversation_history}

## Current Interaction/Question/Query
**Current Question:**
{current_question}

## Response Guidelines
Focus on the current question first and use the conversation context and user information as context to answer the question.

## Available Tools
You can use the following tools to help you:
1. **RAG_tool**: To answer questions about any insurance policy
2. **web_search_tool**: To answer questions about any insurance policy if the RAG_tool is not able to answer the question

"""

def answer_question(question: str, state: UserProfile, node_name: str) -> str:
    context = {
        "user_profile": state.get_summary(),
        "recommendations": state.recommeneded_policies,
        "conversation_history": state.messages,
        "current_question": question
    }
    prompt_template = persuasion_prompt

    # Format the template with context
    formatted_prompt = prompt_template.format(**context)

    # Get response from LLM 
    response = agent.invoke({"messages": formatted_prompt})
    response_content = response['messages'][-1].content

    # Add messages as strings, not dictionaries
    state.messages.append(f"User: {question}")
    state.messages.append(f"Assistant: {response_content}")

    return Command(
        goto=node_name,
        update={
            **state.model_dump(),
            "agent_query": response_content,
            "profiling_stage": node_name,
            }
    )

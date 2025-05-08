# Prerequisite functions  and imports for the main functions

from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
# Load environment variables from .env file
load_dotenv()


# ------------------------------------------------------------------------------------------------
# OpenAI LLM
# ------------------------------------------------------------------------------------------------
openai_model = "gpt-4o"
openai_api_key = os.getenv("OPENAI_API_KEY")
llm_openai = ChatOpenAI(temperature=0.0,api_key=openai_api_key,model=openai_model)

# ------------------------------------------------------------------------------------------------
# Gemini LLM
# ------------------------------------------------------------------------------------------------
gemini_model = "gemini-2.0-flash"
gemini_api_key = os.getenv("GOOGLE_API_KEY")
llm_gemini = ChatGoogleGenerativeAI(model=gemini_model,temperature=0.0, google_api_key=gemini_api_key, max_tokens=None)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# ------------------------------------------------------------------------------------------------
# Anthropic LLM
# ------------------------------------------------------------------------------------------------
anthropic_model = "claude-3-5-sonnet-20240620"
anthropic_api_key = os.getenv("X_API_KEY")
llm_anthropic = ChatAnthropic(temperature=0.0,api_key=anthropic_api_key,model=anthropic_model   )

# ------------------------------------------------------------------------------------------------
# Default LLM for all the inner workings of the project
# ------------------------------------------------------------------------------------------------
llm = llm_gemini

gemini_api_key2 = os.getenv("GOOGLE_API_KEY_1")
backup_llm = ChatGoogleGenerativeAI(model=gemini_model,temperature=0.0, google_api_key=gemini_api_key2, max_tokens=None)


from typing import Literal,List
from pydantic import BaseModel, Field

class RoutePolicy(BaseModel):
    policy_name: List[Literal[
        "Bajaj_Allianz_Network_List",
        "CARE-PLUS",
        "Bajaj_My Health Care",
        "Care_Ultimate_Care",
        "Tata_AIG_Network_List",
        "StarHealth_Assure",
        "Care_Freedom",
        "ICICI_Max_Protect_Classic",
        "NivaBupa Rise",
        "NivaBupa_Go_Active",
        "Star_Health_Network_List",
        "ICICI Elevate",
        "ICICI_Health_AdvantEdge",
        "HDFC-optima_secure",
        "AdityaBirla_Activ_Fit",
        "Tata_Aig_Medicare",
        "Care Senior Health Advantage",
        "HDFC_Ergo_Energy_Gold",
        "NivaBupa_Aspire",
        "AdityaBirla_Activ_Health_Platinum_Enhanced",
        "TATA_AIG_Medicare_Plus",
        "NivaBupa_Health_Companion",
        "StarHealth_Cardiac-Care",
        "NivaBupa_Health_Pulse_Enhanced",
        "Galaxy_Promise - Elite",
        "HDFC_Optima_Restore",
        "StarHealth_Comprehensive",
        "Care Joy",
        "NivaBupa_Health_Premia",
        "Care Supreme Combo",
        "NivaBupa Health ReAssure",
        "Care_Network_List",
        "AdityaBirla_Activ Health Platinum Essential",
        "AdityaBirla_activ_one",
        "ICICI_MaxProtect",
        "Care Supreme Senior Premium",
        "NivaBupa_ReAssure",
        "ICICI_Supertopup_healthbooster",
        "Star Health_Super Star",
        "StarHealth_Smart_Health_Pro",
        "StarHealthMediClassic",
        "ReAssure 2.0 bronze Plus",
        "Star_Health_Young Star Gold Plan",
        "Care Senior",
        "NivaBupa_Aspire_Gold",
        "CARE Heart",
        "Care_Advantage",
        "NivaBupa_Arogya_Sanjeevani",
        "CARE SUPREME",
        "TATA_AIG_Medicare_Lite"
    ]] = Field(
        ...,
        description="List of applicable insurance policies for the user query"
    )


# ------------------------------------------------------------------------------------------------
# Supabase Vector Store
# ------------------------------------------------------------------------------------------------

from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from supabase.client import Client, create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
# print(SUPABASE_URL, SUPABASE_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ ERROR: Supabase credentials are missing! Check your .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

vectorstore = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

# ------------------------------------------------------------------------------------------------  
# Summaries Vector Store
# ------------------------------------------------------------------------------------------------

summaries_vectorstore = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="one_page_summaries",
    query_name="ops_match_documents",
)


# ------------------------------------------------------------------------------------------------
# Query classification
# ------------------------------------------------------------------------------------------------
from pydantic import BaseModel, Field

class QueryResponse(BaseModel):
    # Purpose: This class defines the structure for classifying user queries about insurance
    # Why: To standardize query classification and enable appropriate response handling
    # How: Uses Pydantic BaseModel to validate query types and ensure consistent categorization
    # The query_type field will be populated by the LLM to classify incoming user queries
    # into predefined categories for routing to the appropriate response handler
    query_type: str = Field(
        description=(
            "Classifies the query into one of the following categories:\n"
            "- policy_recommendation_request: If the user is asking for policy recommendations\n" 
            "- service_information_request: If the user is asking for information about services like claims, renewals, etc.\n"
            "- policy_information_request: If the user is asking for details about specific policiy details\n"
            "- insurer_information_request: If the user is asking for information about insurance providers like HDFC ERGO, ICICI Lombard, etc."
            "- policy_comparison_request: If the user is asking to compare different insurance policies\n"
            "- analysis_request: If the user is asking for deep analysis on any health insruance related topic, be it recommendations, comparison, validation etc.\n"
            "- other: If the user's query does not fit into the above categories"
        )
    )



import os
import sys
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))


import os
from dotenv import load_dotenv
import random

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
os.environ['LANGCHAIN_TRACING'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGSMITH_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = 'GAIDO-EXP'

from functions.prereq import llm, embeddings
# from functions.prereq import vectorstore
# from functions.fx import RAG_tool
from typing import Literal


from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from my_agent.user_state import UserProfile

from functions.fx import answer_question, llm_with_tools


engaging_tips = [
    "💡 Buying health insurance at 25 can cost 50% less than at 40. Start early, save big.",
    "🛡️ A single hospital visit can cost more than 1 year of insurance premiums. Prevention pays.",
    "📲 Most modern insurers now offer cashless treatment through mobile apps—zero paperwork!",
    "📈 Each year you skip insurance, your premium rises. It's cheaper to start now.",
    "🏥 Not all hospital rooms are covered—room rent caps can cost you. Always check before admitting.",
    "👨‍👩‍👧‍👦 Family floater plans are perfect if everyone's healthy. Shared coverage = shared savings.",
    "🚨 Missed your policy renewal? You could lose lifetime benefits. Set that reminder!",
    "💰 You can save up to ₹75,000 in taxes (India) just by having health insurance under Section 80D.",
    "🎁 No-claim bonuses (NCB) can boost your coverage by 50% or more—just for staying healthy!",
    "🧘‍♂️ Mental health is now covered in many policies. Therapy, counseling, psychiatry—it's all in.",
    "🧾 Always ask for a detailed bill. Insurers need itemized invoices for smooth claim processing.",
    "👶 Maternity cover has a waiting period (2–4 years)—plan in advance if you're planning a family.",
    "⚖️ Co-pay means you split the bill. Go for policies with lower or zero co-pay.",
    "🎉 Some plans reward healthy habits like walking or gym visits. Earn wellness points = discounts!",
    "🌐 You can port your insurance plan if you're unhappy—just like switching SIM cards!",
    "🔐 Don't hide pre-existing conditions. Transparency = guaranteed claims in the future.",
    "💳 Always carry your e-health card. Emergencies don't wait for documents.",
    "💡 The earlier you buy, the shorter your waiting period ends. Time is coverage.",
    "🔍 A ₹10 lakh cover today might not be enough 10 years from now—revisit coverage annually.",
    "🛌 Sub-limits on surgeries, room rent, or ICU can ruin your finances. Read that fine print!",
    "🏆 Premium ≠ Best. Don't judge a plan just by price—benefits are what matter.",
    "🧑‍⚕️ Some policies now cover OPD visits, dental care, and alternative treatments like Ayurveda.",
    "👀 Many people never use insurance... but when you need it, it can save you ₹5L+ in one go.",
    "📆 Claim rejection is highest when people hide conditions or miss deadlines. Don't be that person.",
    "📝 You can buy insurance online in 10 mins. No agents, no delays, just peace of mind.",
    "🏃 Healthier you = lower risk profile = better premium offers. Stay fit, save money.",
    "💬 Some insurers offer WhatsApp support and instant claim updates. Customer service, upgraded!",
    "🔁 Even if you haven't claimed in years, insurance is your safety net. Don't drop it.",
    "🤖 AI is now used by insurers to settle claims faster. 2-min decisions are becoming real!",
    "🌍 Health insurance can cover international treatments too—if you choose global coverage."
]

def get_tips():
    return random.choice(engaging_tips)

def greeting_node(state: UserProfile) -> Command[Literal["personal_info"]]:
    """Initial greeting node with empathetic introduction."""
    print('in greeting node')
    
    # If greeting is already done, just move to next node
    if state.greeting_done:
        return Command(
            goto="personal_info",
            update={
                "interaction_count": state.interaction_count + 1
            }
        )
    
    # Generate a warm, personalized response
    response = llm.invoke(f"""
    Craft a warm, personalized greeting for Gaido that:
    - Uses the user's name from login_context (e.g., "Hey {state.name}!")
    - Sounds human, caring, and empathetic
    - Briefly explains Gaido's onboarding process
    - Mentions how Gaido can help (e.g., health tracking, doctor support, insurance guidance, etc.)
    - Makes the user feel welcomed and supported
    - Give only one greeting

    login_context: Name: {state.name}
    example: Hey Sadikh Welcome to Gaido 😊 I'm here to make your onboarding simple and stress-free. We'll guide you through everything — from health tracking and doctor consultations to insurance support. You're in caring hands!
    If the user is not logged in, use the following greeting:
    example: Hey! Welcome to Gaido 😊 I'm here to make your onboarding simple and stress-free. We'll guide you through everything — from health tracking and doctor consultations to insurance support. You're in caring hands!

""")

    
    print(response.content)
    state.messages.append(response.content)
    return Command(
        goto="personal_info",
        update={
            "messages": [response.content],
            "greeting_done": True,
            "interaction_count": state.interaction_count + 1,
            "profiling_stage": "greeting"
        }
    )

def personal_info_node(state: UserProfile) -> Command[Literal["health_info","personal_info"]]:
    """Collect all personal information before proceeding."""
    print(f"state: {state.model_dump()}")
    # Check if we have all required info
    has_all_info = all([
        state.name,
        state.age,
        state.contact_info
    ])
    
    # If we have all info, move to random fact
    if has_all_info:
        return Command(
            goto="health_info",
            update={
                "personal_info_collected": True,
                "profiling_stage": "complete",
                "interaction_count": state.interaction_count + 1,
            }
        )
    
    # Otherwise, collect missing information one at a time with validators
    if not state.name:
        # Name validation
        while True:
            # name_query = llm.invoke("Give me a simple question asking User's name").content
            name_query ="""Before we take care of your health insurance, let's get to know you! 👋

            What should we call you?"""
            name_query = state.agent_query +"\n"+ name_query if state.agent_query else name_query
            name = interrupt(name_query)
            response = llm_with_tools.invoke(name)
            if len(response.tool_calls) > 0:
                return answer_question(name, state, "personal_info")
            # Validate name (non-empty, contains only letters and spaces)
            if not name or name.isspace():
                validation_msg = interrupt("Name cannot be empty. Please provide your name:")
                continue
            
            if any(char.isdigit() for char in name):
                validation_msg = interrupt("Name should not contain numbers. Please provide a valid name:")
                continue
                
            # Valid name - proceed
            return Command(
                goto="personal_info",  # Loop back to same node
                update={
                    "name": name,
                    "profiling_stage": "name",
                    "interaction_count": state.interaction_count + 1
                }
            )
        
    if not state.family_members:
    # Family members validation
        while True:
            family_query = """
            Who would you like to include in your health insurance coverage? 🏥👨‍👩‍👧‍👦

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

            Please list all family members you'd like to insure:
            """
            family_input = interrupt(family_query)
            response = llm_with_tools.invoke(family_input)
            if len(response.tool_calls) > 0:
                return answer_question(family_input, state, "personal_info")
            
            # Basic validation - non-empty response
            if not family_input or family_input.isspace():
                validation_msg = interrupt("Please provide information about who you want to insure:")
                continue
                
            # Convert response to list and clean up
            family_members = [member.strip() for member in family_input.lower().replace('and', ',').split(',')]
            family_members = [m for m in family_members if m and not m.isspace()]
            
            if not family_members:
                validation_msg = interrupt("Could not understand who you want to insure. Please try again:")
                continue
            # Valid input - proceed
            return Command(
                goto="personal_info",  # Loop back to same node
                update={
                    "family_members": family_members,
                    "recent_state": "family_members",
                    "interaction_count": state.interaction_count + 1
                }
            )
    
    
    if not state.age:
        # Age validation
        while True:
            # age_query = llm.invoke("Give me a simple question asking User's age").content
            age_query = f"""
            Please provide the age for each family member you listed ({', '.join(state.family_members)}):

            How to enter:
            Enter ages in the same order, separated by commas. For example:
            "35, 32, 5" 

            Please enter the ages now:
            """
            age_input = interrupt(age_query)
            response = llm_with_tools.invoke(age_input)
            if len(response.tool_calls) > 0:
                return answer_question(age_input, state, "personal_info")
            
            # Parse the ages into a list
            try:
                ages = [int(age.strip()) for age in age_input.split(',')]
                
                # Validate each age
                if len(ages) != len(state.family_members):
                    validation_msg = interrupt(f"Please provide ages for all {len(state.family_members)} family members:")
                    continue
                    
                if any(age < 0 or age > 120 for age in ages):
                    validation_msg = interrupt("Please provide valid ages between 0 and 120:")
                    continue
                    
                # Valid ages - proceed
                return Command(
                    goto="personal_info",  # Loop back to same node
                    update={
                        "age": ages,
                        "recent_state": "age",
                        "interaction_count": state.interaction_count + 1
                    }
                )
            except ValueError:
                validation_msg = interrupt("Ages must be numbers. Please provide ages as numbers separated by commas:")
                continue
    
    # Contact information validation
    while True:
        # contact_info_query = llm.invoke("Give me a simple question asking User's contact information like email or phone number").content
        contact_info_query = """
        To help us stay in touch, please share your preferred contact information:
        - Email address, or 
        - Phone number
        
        This helps us keep you updated about your insurance journey."""
        contact_info = interrupt(contact_info_query)
        response = llm_with_tools.invoke(contact_info)
        if len(response.tool_calls) > 0:
            return answer_question(contact_info, state, "personal_info")

        # Validate contact info (must contain @ for email or be a valid phone format)
        if not contact_info or contact_info.isspace():
            validation_msg = interrupt("Contact information cannot be empty. Please provide your email or phone number:")
            continue
            
        # Simple email validation - contains @ and .
        is_email = '@' in contact_info and '.' in contact_info
        
        # Simple phone validation - contains enough digits
        digits_count = sum(1 for char in contact_info if char.isdigit())
        is_phone = digits_count >= 10
        
        if not (is_email or is_phone):
            validation_msg = interrupt("Please provide a valid email address or phone number:")
            continue
            
        tip = get_tips()
        print("DID YOU KNOW?")
        print(tip)
        # Valid contact info - proceed
        return Command(
            goto="health_info",  # Loop back to same node
            update={
                "contact_info": contact_info,
                "personal_info_collected": True,
                "profiling_stage": "contact",
                "interaction_count": state.interaction_count + 1
            }
        )
    

def health_info_node(state: UserProfile) -> Command[Literal["onboarding_confirmation","health_info"]]:
    """Collect health information with empathetic and adaptive approach."""
    
    # First check if we have all required health info
    has_all_info = all([
        state.has_pre_existing_conditions is not None,
        (state.pre_existing_conditions if state.has_pre_existing_conditions else True)
    ])
    
    # If we have all info, move to next node
    if has_all_info:
        return Command(
            goto="onboarding_confirmation",
            update={
                "health_info_collected": True,
                "profiling_stage": "health_complete",
                "interaction_count": state.interaction_count + 1
            }
        )
    
    # First ask about pre-existing conditions in general
    if state.has_pre_existing_conditions is None:
        pre_existing_query = f"""Hey {state.name if state.name else "there"}!

        💡 Knowing about any pre-existing medical conditions in the family helps us recommend plans that offer the right benefits—like shorter waiting periods, better coverage, or specialized care.

        For families with multiple members, it's important to choose insurance that fits everyone's health needs—whether it's for a child, parent, or senior.

        👉 To make sure we get this right, could you let us know if **you or any family members** 
        have any pre-existing medical conditions?
        (Examples: diabetes, heart conditions, high blood pressure)

        Just reply with **yes** or **no**. """

        has_conditions = interrupt(pre_existing_query)
        
        # Check for tool calls
        response = llm_with_tools.invoke(has_conditions)
        if len(response.tool_calls) > 0:
            return answer_question(has_conditions, state, "health_info")
            
        # Update state based on yes/no response
        has_pre_existing = has_conditions.lower().startswith('y')
        
        if not has_pre_existing:
            # If no pre-existing conditions, mark as complete
            return Command(
                goto="onboarding_confirmation",
                update={
                    "has_pre_existing_conditions": False,
                    "pre_existing_conditions": [],
                    "health_info_collected": True,
                    "profiling_stage": "health_complete",
                    "interaction_count": state.interaction_count + 1
                }
            )
        else:
            # If they have conditions, update state and continue to collect details
            return Command(
                goto="health_info",
                update={
                    "has_pre_existing_conditions": True,
                    "profiling_stage": "pre_existing_check",
                    "interaction_count": state.interaction_count + 1
                }
            )
    
    # If they have pre-existing conditions, collect details
    if state.has_pre_existing_conditions and not state.pre_existing_conditions:
        
        conditions_query = f"""
        Thank you for sharing that {state.name} 💙 Understanding your specific health conditions helps us match you with insurance plans that offer the right support—like shorter waiting periods, 
        specialized coverage, or chronic care benefits.

        Could you please list the pre-existing conditions for yourself or any family members?
        (Example: diabetes, high blood pressure, asthma)

        👉 Just separate them with commas so we can guide you better.
        """
        conditions = interrupt(conditions_query)
        
        # Check for tool calls
        response = llm_with_tools.invoke(conditions)
        if len(response.tool_calls) > 0:
            return answer_question(conditions, state, "health_info")
            
        # Parse conditions into a list, handling empty input
        if conditions and not conditions.isspace():
            condition_list = [c.strip() for c in conditions.split(',') if c.strip()]
        else:
            condition_list = []
        
        return Command(
            goto="onboarding_confirmation",
            update={
                "pre_existing_conditions": condition_list,
                "health_info_collected": True,
                "profiling_stage": "health_complete",
                "interaction_count": state.interaction_count + 1
            }
        )
    
    # If we reach here, we should have all required info
    return Command(
        goto="onboarding_confirmation",
        update={
            "health_info_collected": True,
            "profiling_stage": "health_complete",
            "interaction_count": state.interaction_count + 1
        }
    )

from langgraph.graph import StateGraph, START, END

def onboarding_confirmation_node(state: UserProfile) -> Command[Literal["personal_info","__end__"]]:
    """Confirm collected information and allow corrections."""

    summary = llm.invoke(f"""
    Create a detailed, friendly summary of the user's information in a clear format:
    - Name: {state.name}
    - Family Members: {state.family_members}
    - Age of family members: {state.age}
    - Contact Info: {state.contact_info}
    - Has Pre-existing Conditions: {state.has_pre_existing_conditions}
    - Pre-existing Conditions: {', '.join(state.pre_existing_conditions) if state.pre_existing_conditions else 'None'}
   
    
    Keep the tone warm and professional.
    """)
    
    # Ask for confirmation
    confirmation = interrupt(f"{summary.content}\n\nIs this information correct? Type 'yes' to proceed or 'no' to make corrections.")
    
    # Check for tool calls
    response = llm_with_tools.invoke(confirmation)
    if len(response.tool_calls) > 0:
        return answer_question(confirmation, state, "onboarding_confirmation")
        
    if confirmation.lower().strip() == 'yes':
        # If confirmed, end the process
        return Command(
            goto=END,
            update={
                **state.model_dump(),
                "onboarding_confirmation_done": True,
                "onboarding_complete": True,
                "profiling_stage": "complete",
                "interaction_count": state.interaction_count + 1,
                "messages": [summary.content, "Thank you for confirming your information!"]
            }
        )
    else:
        # If not confirmed, reset relevant flags and restart from profile collection
        return Command(
            goto="personal_info",
            update={
                **state.model_dump(),
                "personal_info_collected": False,
                "health_info_collected": False,
                "onboarding_confirmation_done": False,
                "has_pre_existing_conditions": None,
                "pre_existing_conditions": [],
                "profiling_stage": "restart",
                "interaction_count": state.interaction_count + 1,
                "messages": ["Let's collect your information again to ensure everything is accurate."]
            }
        )
    
onboarding_workflow = StateGraph(UserProfile)

# Add nodes
onboarding_workflow.add_node("greeting", greeting_node)
onboarding_workflow.add_node("personal_info", personal_info_node)
onboarding_workflow.add_node("health_info", health_info_node)
onboarding_workflow.add_node("onboarding_confirmation", onboarding_confirmation_node)
# onboarding_workflow.add_node("sessions", sessions_node)  # Add sessions node if needed

# Set entry point and basic edges
onboarding_workflow.set_entry_point("greeting")
onboarding_workflow.set_finish_point("onboarding_confirmation")
config = {"configurable": {"thread_id": "unique_conversation_id"}}
onboarding_graph = onboarding_workflow.compile()
# onboarding_graph = onboarding_workflow.compile(checkpointer=MemorySaver())

# onboarding_graph.invoke({"name": "Sadique", "age": 25, "contact_info": "sadique@gmail.com"},config=config)
# state = onboarding_graph.get_state(config=config)
# print(state.values)

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
class UserProfile(BaseModel):
    
    # Personal info details
    name: Optional[str] = Field(default=None)
    family_members: List[str] = Field(default_factory=list)
    age: Optional[List[int]] = Field(default_factory=list)
    agent_query: Optional[str] = Field(default=None)
    contact_info: Optional[str] = Field(default=None)
    pre_existing_conditions: List[str] = Field(default_factory=list)
    has_pre_existing_conditions: Optional[bool] = None
    user_intent_query: Optional[str] = Field(default=None)
    
    messages: List[str] = Field(default_factory=list)
    

    # Preferences Collected
    preferences_data: List[Dict[str, Optional[str]]] = Field(default_factory=list)
    current_question_index: int = 0
    
    coverage_type: Optional[str] = None  # Individual, Family, Senior Citizen, Corporate
    budget_range: Optional[str] = None  # Low, Medium, High, or exact budget
    # specific_benefits: List[str] = Field(default_factory=list)  # Maternity, Dental, etc.
    specific_benefits: Optional[str] = None  # Maternity, Dental, etc.
    user_query: Optional[str] = None 
    recommeneded_policies: Optional[str] = None 

    # Interaction and Flow Control
    interaction_count: int = 0
    profiling_stage: Optional[str] = None
    greeting_done: bool = False
    personal_info_collected: bool = False
    health_info_collected: bool = False
    preferences_collected: bool = False
    policy_match_done: bool = False
    query_handling_done: bool = False
    onboarding_confirmation_done: bool = False
    recommendation_confirmation_done: bool = False
    completion_done: bool = False
    
    current_workflow: str = "onboarding"  # Track which workflow is active
    
    # Boolean Flags for User State
    
    new_user: bool = True

    # Status Flags
    onboarding_complete: bool = False
    recommendation_complete: bool = False


    def has_missing_profile_info(self) -> bool:
        """
        Checks if any critical user profile information is missing.
        Returns True if any key information is missing, False if all required info is present.
        """
        # Check if name is missing
        if not self.name:
            return True
            
        # Check if family members list is empty
        if not self.family_members:
            return True
            
        # Check if age list is missing or doesn't match family members length
        if not self.age or len(self.age) != len(self.family_members):
            return True
            
        # Check if health conditions are collected
        if self.has_pre_existing_conditions is None:
            return True
        
        if self.has_pre_existing_conditions and not self.pre_existing_conditions:
            return True
        
        return False
    
    def return_missing_profile_info(self) -> List[str]:
        """
        Returns a list of missing critical user profile information fields.
        """
        missing_fields = []
        
        # Check if name is missing
        if not self.name:
            missing_fields.append("name")
            
        # Check if family members list is empty
        if not self.family_members:
            missing_fields.append("family_members")
            
        # Check if age list is missing or doesn't match family members length
        if not self.age:
            missing_fields.append("age")
        elif len(self.age) != len(self.family_members):
            missing_fields.append("age (incomplete for all family members)")
            
        # Check if health conditions are collected
        if self.has_pre_existing_conditions is None:
            missing_fields.append("has_pre_existing_conditions")
        elif self.has_pre_existing_conditions and not self.pre_existing_conditions:
            missing_fields.append("pre_existing_conditions")
            
        return missing_fields
    
    def update_state(self, name: Optional[str] = None, 
                    age: Optional[List[int]] = None,
                    family_members: Optional[List[str]] = None,
                    has_pre_existing_conditions: Optional[bool] = None,
                    pre_existing_conditions: Optional[List[str]] = None) -> None:
        """
        Updates the user state with provided information.
        Only updates fields that are provided (not None).
        """
        if name is not None:
            self.name = name

        # Update age properly - don't append lists, extend or assign
        if age is not None:
            if isinstance(age, list):
                if not self.age:
                    self.age = age
                else:
                    self.age.extend(age)
            else:
                self.age.append(age)

        # Update family members properly - don't append lists, extend or assign
        if family_members is not None:
            if isinstance(family_members, list):
                if not self.family_members:
                    self.family_members = family_members
                else:
                    self.family_members.extend(family_members)
            else:
                self.family_members.append(family_members)
        
        if has_pre_existing_conditions is not None:
            self.has_pre_existing_conditions = has_pre_existing_conditions

        # Update pre-existing conditions properly - don't append lists, extend or assign
        if pre_existing_conditions is not None:
            if isinstance(pre_existing_conditions, list):
                if not self.pre_existing_conditions:
                    self.pre_existing_conditions = pre_existing_conditions
                else:
                    self.pre_existing_conditions.extend(pre_existing_conditions)
            else:
                self.pre_existing_conditions.append(pre_existing_conditions)

    def get_summary(self) -> str:
        """
        Returns a formatted summary of the user profile information.
        """
        summary_parts = []
        
        # Always show name status
        summary_parts.append(f"Name: {self.name}")
            
        # Always show family members status
        if self.family_members:
            flat_family_members = []
            for item in self.family_members:
                if isinstance(item, list):
                    flat_family_members.extend(item)
                else:
                    flat_family_members.append(item)
            summary_parts.append(f"Family Members: {', '.join(str(member) for member in flat_family_members)}")
        else:
            summary_parts.append("Family Members: None")
            
        # Always show age status    
        if self.age:
            flat_ages = []
            for item in self.age:
                if isinstance(item, list):
                    flat_ages.extend(item)
                else:
                    flat_ages.append(item)
            summary_parts.append(f"Age: {', '.join(str(age) for age in flat_ages)}")
        else:
            summary_parts.append("Age: None")
            
        
        
        # Always show pre-existing conditions status
        if self.pre_existing_conditions:
            flat_conditions = []
            for item in self.pre_existing_conditions:
                if isinstance(item, list):
                    flat_conditions.extend(item)
                else:
                    flat_conditions.append(item)
            summary_parts.append(f"Pre-existing Conditions: {', '.join(str(condition) for condition in flat_conditions)}")
        else:
            summary_parts.append("Pre-existing Conditions: None")
            
        if self.preferences_data:
            flat_preferences = []
            for item in self.preferences_data:
                if isinstance(item, (list, dict)):
                    if isinstance(item, dict):
                        flat_preferences.extend(f"{k}: {v}" for k, v in item.items())
                    else:
                        flat_preferences.extend(item)
                else:
                    flat_preferences.append(str(item))
            summary_parts.append(f"Preferences: {', '.join(flat_preferences)}")
        else:
            summary_parts.append("Preferences: None")
            
        return "\n".join(summary_parts)
    
    def get_summary_with_messages_and_recommendations(self) -> str:
        """
        Returns a formatted summary of the user profile information.
        """
        summary_parts = []
        summary_parts.append(self.get_summary())
        
        if self.messages:
            messages_str = "\nConversation History:\n"
            for msg in self.messages:
                messages_str += f"- {msg}\n"
            summary_parts.append(messages_str)
            
        if self.recommeneded_policies:
            reco_str = "\nRecommended Policies:\n"
            for policy in self.recommeneded_policies:
                reco_str += f"- {policy}\n"
            summary_parts.append(reco_str)
            
        return "\n".join(summary_parts)
 










# Multi Agent Framework Test Case Scenarios

## Framework Overview

The Multi Agent Framework consists of the following actual components:

1. **Core Agents**:

   a. **Onboarding Agent** (`onboarding_agent.py`):
      - Greeting Node: Handles initial user greeting and introduction
      - Personal Info Node: Collects user's name, age, and contact information
      - Health Info Node: Gathers health-related information
      - Onboarding Confirmation Node: Finalizes the onboarding process

   b. **Recommendation Agent** (`recommendation_agent.py`):
      - Handles policy recommendations based on user profile
      - Processes user queries about insurance policies
      - Provides personalized insurance recommendations

   c. **Supervisor Node** (`supervisor_node.py`):
      - Coordinates workflow between agents
      - Manages state transitions
      - Handles user intent routing
      - Processes different query types (policy comparison, recommendation, information)

   d. **Policy Comparison Node** (`policy_comparison_node.py`):
      - Compares different insurance policies
      - Evaluates policy features and benefits

   e. **Policy Info Node** (`policy_info_node.py`):
      - Provides detailed information about specific policies
      - Answers policy-related queries

   f. **Fallback Node** (`fallback_node.py`):
      - Handles error cases and edge scenarios
      - Provides fallback responses

2. **State Management**:
   - User State (`user_state.py`): Manages user profile and preferences
   - Profile Updates (`profile_update.py`): Handles profile modifications
   - Preferences (`preferences.py`): Manages user preferences
   - Follow-up (`follow_up.py`): Handles follow-up interactions

## Test Case Categories

### 1. Onboarding Flow Tests
- **TC1.1**: Greeting Flow
  - Input: New user session
  - Expected: Appropriate greeting based on user state
  - Success Criteria: Greeting message generated, state updated

- **TC1.2**: Personal Information Collection
  - Input: User responses to personal info queries
  - Expected: Validated personal information stored
  - Success Criteria: Name, age, contact info properly stored

- **TC1.3**: Health Information Collection
  - Input: User health-related information
  - Expected: Health profile created
  - Success Criteria: Health information properly stored

### 2. Recommendation System Tests
- **TC2.1**: Policy Recommendation
  - Input: User profile and query
  - Expected: Relevant policy recommendations
  - Success Criteria: Recommendations match user profile

- **TC2.2**: Query Processing
  - Input: User queries about policies
  - Expected: Appropriate responses
  - Success Criteria: Responses address user queries

### 3. Policy Management Tests
- **TC3.1**: Policy Comparison
  - Input: Multiple policies to compare
  - Expected: Clear comparison results
  - Success Criteria: Differences and similarities highlighted

- **TC3.2**: Policy Information
  - Input: Policy-related queries
  - Expected: Detailed policy information
  - Success Criteria: Information is accurate and complete

### 4. Error Handling Tests
- **TC4.1**: Fallback Scenarios
  - Input: Invalid or unexpected user input
  - Expected: Graceful error handling
  - Success Criteria: System remains stable

- **TC4.2**: Edge Cases
  - Input: Edge case scenarios
  - Expected: Appropriate handling
  - Success Criteria: System handles edge cases properly

### 5. State Management Tests
- **TC5.1**: Profile Updates
  - Input: Profile modification requests
  - Expected: Profile updated correctly
  - Success Criteria: Changes reflected in user state

- **TC5.2**: Preference Management
  - Input: Preference changes
  - Expected: Preferences updated
  - Success Criteria: New preferences stored correctly

## Test Data Requirements

1. **User Profiles**:
   - Complete profiles
   - Partial profiles
   - Edge case profiles

2. **Policy Data**:
   - Different policy types
   - Various coverage options
   - Different premium ranges

3. **Query Types**:
   - Policy comparison requests
   - Recommendation requests
   - Information requests

## Test Environment Setup

1. **Required Components**:
   - Python environment
   - Required dependencies (from requirements.txt)
   - Test database
   - Environment variables

2. **Configuration**:
   - API keys
   - Environment settings
   - Test data initialization

## Test Execution Guidelines

1. **Pre-test Setup**:
   - Initialize test environment
   - Load test data
   - Configure logging

2. **Test Execution**:
   - Run tests sequentially
   - Monitor system behavior
   - Record results

3. **Post-test Verification**:
   - Validate results
   - Clean up test data
   - Document findings

## Expected Outcomes

1. **Success Criteria**:
   - All test cases pass
   - System remains stable
   - Expected behavior observed

2. **Failure Handling**:
   - Clear error messages
   - System recovery
   - Logging of issues

## Reporting

1. **Test Results**:
   - Pass/fail status
   - Performance metrics
   - Error logs

2. **Documentation**:
   - Test case descriptions
   - Setup instructions
   - Known issues 
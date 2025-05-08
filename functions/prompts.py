from langchain_core.prompts import PromptTemplate


# Query_Rephrase_template = '''
# Imagine you are an advanced conversational AI with deep contextual understanding.
# Your task is to rephrase the user's query based on the provided conversation history while preserving its original intent and meaning. 
# Ensure that the rephrased query is clear, concise, and naturally worded.

# Important Instructions:
# - Each utterance/dialogue in the conversation history is delimited by #### for better understanding.
# - Do not hallucinate or introduce any additional details that are not explicitly mentioned in the conversation history.
# - The rephrased query should stay as close as possible to the original user intent without adding unnecessary specifics.
# - If the user query is vague (e.g., "tell me more about it"), ensure that the rephrased query remains general and does not assume a specific topic unless clearly indicated in the conversation history.

# Conversation History:
# {conv_memory}

# User Query: {Question}

# Your Rephrased Query:
# '''

# Query_Rephrase_prompt_template = PromptTemplate.from_template(Query_Rephrase_template)


# ------------------------------------------------------------------------------------------------
# Feature Guidelines:
# ------------------------------------------------------------------------------------------------

feature_guidelines = """

## Feature Recommendation Guide

This guide helps recommend health insurance features based on different user profiles.

**Category:** Life stage based

**Situation/ Persona:** Young Single, <25, no health issues

**Needs:** Affordable, basic coverage + wellness

**Must have Features:**
* Min coverage of 5-10 lakh based on cumulative bonus
* Good cumulative bonus benefits
* Age lock/“freeze premium” benefit

**Good to have features:**
* Access to Gyms
* Wellness Discount

---

**Category:** Life stage based

**Situation/ Persona:** Young couple just started their family. Planning for kids in near or short term

**Needs:** Maternity & family planning needs

**Must have Features:**
* Min 10 lakh family floater cover
* Maternity & newborn baby cover
* Short maternity waiting period
* Good Restoration benefit (especially for floaters)
* Good cumulative bonus/ no-claim bonus benefits

**Good to have features:**
* Wellness Discount

---

**Category:** Life stage based

**Situation/ Persona:** Families with kids

**Needs:** Shared coverage, continuity of cover

**Must have Features:**
* Min 10 lakh family floater cover
* Good Restoration benefit (especially for floaters)
* Good cumulative bonus/ no-claim bonus benefits

**Good to have features:**
* OPD for kids
* Vaccination cover
* Wellness Discount

---

**Category:** Life stage based

**Situation/ Persona:** Middle age couples

**Needs:** Higher coverages, Preventive care, early detection

**Must have Features:**
* Min 10 lakh family floater cover
* Good Restoration benefit (especially for floaters)
* Good cumulative bonus/ no-claim bonus benefits
* Preventive health checkups

**Good to have features:**
* Critical illness rider (based on family history)
* Wellness Discount

---

**Category:** Life stage based

**Situation/ Persona:** Elderly Parents

**Needs:** Immediate and inclusive cover

**Must have Features:**
* Min 20 lakh family floater cover
* Good Restoration benefit (especially for floaters)
* Good cumulative bonus/ no-claim bonus benefits
* Short waiting period for pre-existing conditions (if any)

**Good to have features:**
* Domiciliary (home) care
* Preventive health checkups
* Wellness Discount

---

**Category:** Life stage based

**Situation/ Persona:** Senior Citizens

**Needs:** Immediate coverage, ease of use

**Must have Features:**
* Min 20 lakh family floater cover
* Good Restoration benefit (especially for floaters)
* Good cumulative bonus/ no-claim bonus benefits
* Short waiting period for pre-existing conditions (if any)
* No pre-medical policy
* Day 1 PED Coverage

**Good to have features:**
* Domiciliary (home) care
* Preventive health checkups

---

**Category:** Health conditions based

**Situation/ Persona:** Chronic Illness (Diabetes, BP, Hypertension, Asthma)

**Needs:** Chronic disease management

**Must have Features:**
* Min 20 lakh cover (Individual/ Floater)
* Good Restoration benefit (especially for floaters)
* Good cumulative bonus/ no-claim bonus benefits
* Short waiting period for pre-existing conditions
* Day 1 PED coverage

**Good to have features:**
* Organ donor coverage
* Chronic care program
* Health checkups

---

**Category:** Health conditions based

**Situation/ Persona:** Cardiac Disease Patients

**Needs:** Targeted cardiac care

**Must have Features:**
* Cardiac-specific plans
* Min 20 lakh cover (Individual/ Floater)
* Good Restoration benefit (especially for floaters)
* Good cumulative bonus/ no-claim bonus benefits
* Short waiting period for pre-existing conditions
* Day 1 PED coverage

**Good to have features:**
* Organ donor coverage
* Cardiac OPD/IPD and follow-up coverage
* Chronic care program
* Health checkups

---

**Category:** Health conditions based

**Situation/ Persona:** High risk applicants

**Needs:** Coverage despite medical history

**Must have Features:**
* Min 20 lakh cover (Individual/ Floater)
* Good Restoration benefit (especially for floaters)
* Good cumulative bonus/ no-claim bonus benefits
* Short waiting period for pre-existing conditions

**Good to have features:**
* Organ donor coverage
* No or minimal pre-medical checkups
* Low or no co-pay
* Higher entry age allowed
* Chronic care program
* Health checkups

---

**Category:** Health conditions based

**Situation/ Persona:** Family history of critical illness

**Needs:** Preparedness for major events

**Must have Features:**
* Min 20 lakh cover (Individual/ Floater)
* Good Restoration benefit (especially for floaters)
* Good cumulative bonus/ no-claim bonus benefits
* Preventive Health checkups

**Good to have features:**
* Lump sum payout for treatment costs
* Critical illness rider

---

**Category:** Health conditions based

**Situation/ Persona:** Mental Health Support seeker

**Needs:** Mental wellness and psychiatry coverage (Features based on lifestage and health)

**Must have Features:**
* Mental health OPD & IPD coverage
* Therapy reimbursement

---

**Category:** Psychology based

**Situation/ Persona:** Affordability focussed or price sensitive

**Needs:** Maximize value at low cost (Prices, Discounts)

**Must have Features:**
* Suitable cover (based on life stage and health)
* Good Restoration benefit (especially for floaters)
* Good cumulative bonus/ no-claim bonus benefits
* Lower premiums

**Good to have features:**
* Discounts
* Wellness Discount
* Tenure Discount

---

**Category:** Psychology based

**Situation/ Persona:** Want great coverage without high costs or prefers unlimited coverage with flexibility, Financially savvy

**Needs:** High, flexible coverage at low cost

**Must have Features:**
* Suitable cover (based on life stage and health)
* Good Restoration benefit (especially for floaters)
* High cumulative bonus/ no-claim bonus benefits
* Lower premiums

**Good to have features:**
* Discounts
* Wellness Discount
* Tenure Discount
* Other features (based of life stage and health)
* Consumables cover

---

**Category:** Psychology based

**Situation/ Persona:** Comprehensive coverage seekers

**Needs:** Want the best of features in everything

**Must have Features:**
* Suitable cover (based on life stage and health)
* Good Restoration benefit (especially for floaters)
* High cumulative bonus/ no-claim bonus benefits

---

**Category:** Psychology based

**Situation/ Persona:** Peace of mind, driven by claims anxiety, lacks trust in insurers

**Needs:** Claims guarantee, Claims support, Minimal hassle (Features based on lifestyle and health)

**Must have Features:**
* Good Claims track record
* Good network hospitals in his area

---

**Category:** Psychology based

**Situation/ Persona:** Peace of mind, driven by coverage anxiety

**Needs:** Long-term, high insurance amount (Features based on lifestyle and health)

**Must have Features:**
* Sum insured up to ₹1 Cr or more
* Unlimited Automatic recharge/restoration
* No claim bonus benefits

---

**Category:** Lifestyle focussed

**Situation/ Persona:** Healthy lifestyle Focussed

**Needs:** Rewards for maintaining health

**Must have Features:**
* Suitable cover (based on life stage and health)
* Good Restoration benefit (especially for floaters)
* High cumulative bonus/ no-claim bonus benefits
* Good Wellness discounts for healthy behavior
* Preventive health checkups

---

**Category:** Lifestyle focussed

**Situation/ Persona:** Travel often

**Needs:** Coverage beyond India (Features based on lifestyle and health)

**Must have Features:**
* Global coverage
* International network hospitals
* Emergency evacuation benefits

---

**Category:** Lifestyle focussed

**Situation/ Persona:** Self employed and gig workers

**Needs:** Budget-friendly, flexible payments (Features based on lifestyle and health)

**Must have Features:**
* Affordable premiums

**Good to have features:**
* Flexi-premium or EMI options


"""

__all__ = ['needs_assessment_prompt']

from langchain_core.prompts import ChatPromptTemplate


# ------------------------------------------------------------------------------------------------
# Initial recommendation guidelines
# ------------------------------------------------------------------------------------------------

initial_recommendation_guidelines = """

Use the below policies and their summaries for determining the recommendation


<HDFC Optima Secure:>
1. Key Features: 4x coverage (Base + Secure + Restore + Plus), 2x sum insured from day 1, 100 percent increase after 2 years (regardless of claims), full restore after any claim, zero deductions on non-medical expenses.
2. Missing Features / Drawbacks: 3-year PED waiting with no add-ons, no maternity or fertility cover, high premium (70 to 75 percent more than budget plans).
3. Pricing: Premiums are on the higher side - best suited for those comfortable paying extra for comprehensive benefits.
4. Claims Experience: Excellent with a 98 percent claim settlement ratio -  among the best in the industry.
5. Best Suited For: Young, healthy professionals who want long-term peace of mind, high claim reliability, and don't mind higher premiums.
6. Not Ideal For: Budget-conscious buyers, those needing maternity or fertility cover, or wanting shorter PED waiting periods.
---------------------------------------     

<ICICI Elevate:>
1. Key Features: Unlimited coverage resets with Infinite Reset, unlimited sum insured with Infinite Sum Assured add-on, one-time infinite claim with Infinite Care, 100 percent yearly cumulative bonus with PowerBooster add-on, PED wait reduced to 30 days with Jumpstart (for diabetes, hypertension, asthma, obesity, etc.), wellness rewards offering up to 30 percent renewal discount, inflation-proof coverage with Inflation Protector, and non-payable expenses covered through Claim Protector.
2. Missing Features / Drawbacks: Lower claim settlement ratio compared to top-tier plans, smaller hospital network compared to competitors.
3. Pricing: Economically priced - around 30 to 40 percent cheaper than plans like Optima Secure (even after add-ons).
4. Claims Experience: ICICI Lombard has a claim settlement ratio of 85 percent, which is lower than the recommended benchmark of 90 percent.
5. Best Suited For: Budget-conscious buyers who want affordability, people with pre-existing conditions using Jumpstart to reduce PED wait to 30 days, families and couples seeking maternity or fertility coverage, and those wanting unlimited and flexible coverage with long-term security.
6. Not Ideal For: People who prioritize high claim settlement ratios or need access to a larger hospital network - ICICI Elevate has fewer tie-ups than leading competitors.
---------------------------------------
     
<Aditya Birla Activ Health Platinum Enhanced:>
1. Key Features: Day-one cover for chronic illnesses like diabetes, hypertension, cholesterol, and asthma; automatic upgrade to Chronic Management Program at no extra cost if you develop a chronic condition; earn up to 100 percent HealthReturns by maintaining 13 Activ Days per month; double your sum insured in 2 years with 50 percent no-claim bonus annually; access to expert health coaches for physical, mental, and nutritional guidance; 100 percent sum insured reload for unrelated illnesses in the same year.
2. Missing Features / Drawbacks: No maternity benefits included in the base plan.
3. Pricing: Feature-rich plan with multiple wellness incentives, may not be suited for those seeking simple or budget options.
4. Claims Experience: Not explicitly stated, but the plan emphasizes modern treatments and cashless care.
5. Best Suited For: Health-conscious individuals and families focused on wellness and fitness; people with or at risk for chronic conditions like diabetes or hypertension; young to mid-aged professionals who value digital health tools, preventive care, and lifestyle-linked benefits; users who want flexible, customizable plans with options for OPD, international cover, and more.        
6. Not Ideal For: People not interested in wellness engagement like step tracking or Activ Dayz, users looking for ultra-simple or low-cost plans, and those prioritizing broad maternity or OPD coverage as their primary need.
---------------------------------------
     
<Care Supreme:>
1. Key Features: Up to 600 percent increase in total coverage, unlimited automatic recharge for both related and unrelated illnesses, up to 30 percent renewal discount through wellness tracking, up to 100 percent of sum insured for ambulance cover, and unlimited e-consultations with general physicians.
2. Missing Features / Drawbacks: No maternity cover included in the base plan.          
3. Pricing: Care Supreme is an economical option, priced lower than many other comprehensive plans.
4. Claims Experience: Care receives a relatively higher number of complaints related to claims experience compared to top-tier insurers.
5. Best Suited For: Budget-conscious individuals and families seeking broad coverage at a low price; people with pre-existing conditions like diabetes, hypertension, or asthma who want to reduce waiting periods to 30 days with add-ons; families needing flexible floater options for multiple members; those wanting unlimited restoration and ambulance benefits; users motivated by wellness rewards and premium discounts.
6. Not Ideal For: Users sensitive to insurer reputation for claims service; those seeking built-in maternity coverage; and individuals who want minimal waiting periods without relying on add-ons.
---------------------------------------
     
<HDFC Ergo Energy Plan:>
1. Key Features: Provides day-one coverage for pre-existing conditions like hypertension and diabetes, with no waiting period.
2. Missing Features / Drawbacks: Limited in scope beyond diabetes and hypertension coverage; not ideal for broader needs.
3. Pricing: Premiums are on the higher side compared to general plans.
4. Claims Experience: Excellent track record with a high claim settlement ratio and smooth claim process.
5. Best Suited For: Individuals already diagnosed with diabetes or hypertension who need immediate coverage without a waiting period.
6. Not Ideal For: People without diabetes or hypertension, as the plan is expensive and may not offer optimal value for those without these conditions.   
---------------------------------------
     
<Niva Bupa ReAssure 2.0 / Aspire Titanium:>
1. Key Features: Offers up to 11x coverage with Base, Booster+, ReAssure Forever, and Safeguard add-ons; Booster+ grows coverage up to 10 times the base sum insured for claim-free years; ReAssure Forever gives unlimited sum insured enhancement; Safeguard/Safeguard+ covers non-payable items; wellness benefits and global coverage available through optional add-ons.
2. Missing Features / Drawbacks: Entry-level variants have long maternity waiting periods (up to 48 months), and many attractive features require paid add-ons which increase total cost.
3. Pricing: Premiums can increase significantly due to multiple add-ons; not the most economical without customization.
4. Claims Experience: Not explicitly stated, but Niva Bupa generally maintains a solid claims reputation with fast-track options.
5. Best Suited For: Young professionals and newly married couples planning for parenthood who want maternity, IVF, and newborn coverage; health-conscious individuals looking for renewal discounts and wellness perks; frequent travelers needing global healthcare access; financially savvy users interested in maximizing long-term value through benefits like Booster+, premium locking, and cashback; large families wanting comprehensive, future-ready health coverage.
6. Not Ideal For: Users seeking a simple or budget plan; those planning a family soon but only opting for base variants with long maternity waits; and people looking for a straightforward policy without the need to manage multiple optional add-ons.
---------------------------------------
     
<Tata AIG Medicare Premier:>
1. Key Features: Global cover for planned hospitalizations diagnosed in India, including visa expenses; consumables benefit covering hospitalization-related items like gloves and masks; automatic 100 percent sum insured restoration with multi-year policies allowing multiple restorations; emergency air ambulance for life-threatening conditions; maternity coverage with delivery complications and first-year vaccinations.
2. Missing Features / Drawbacks: Long waiting periods for maternity (4 years) and listed ailments like cataracts and joint replacements (2 years); no notable fast-access OPD or cumulative bonus benefits.
3. Pricing: Not explicitly mentioned, but positioned as a premium feature-rich plan with international and advanced treatment options.
4. Claims Experience: Generally positive, with Tata AIG known for solid claim service and support.
5. Best Suited For: Individuals or families seeking global healthcare access; those planning for maternity and early childcare; users who value advanced features like air ambulance and consumables cover; good fit for frequent travelers and people who want a comprehensive, high-benefit plan and can wait through the initial exclusion periods.
6. Not Ideal For: People who need quick access to maternity or specific treatments within a short time; users prioritizing rapid PED or daycare coverage; or those looking for high cumulative bonus and OPD flexibility early on.
---------------------------------------

<Niva Bupa ReAssure 2.0:>
1. Key Features: Offers up to 11x coverage with Base, Booster+, ReAssure Forever, and Safeguard add-ons; hospitalization coverage starts from 2 hours; includes pre-hospitalization (60 days) and post-hospitalization (180 days); Booster+ grows coverage up to 10x the base sum insured if no claims are made; ReAssure Forever provides unlimited sum insured enhancement; Safeguard/Safeguard+ covers non-payable items; Carry Forward allows unused coverage to roll over; renewal discounts available through step tracking
2. Missing Features / Drawbacks:
3. Pricing: 
4. Claims Experience: Niva Bupa generally offers a reliable claims experience, supported by strong service infrastructure and digital claims options.       
5. Best Suited For: Individuals and families wanting high sum insured coverage with long-term financial security; users looking for modern and AYUSH treatment coverage; health-conscious individuals interested in wellness-linked discounts; and those who value flexible benefits like Booster+, unlimited enhancements, and non-payable item coverage.
6. Not Ideal For: Users seeking simple, low-cost plans without managing multiple benefits or wellness programs; or those who prefer upfront maternity or OPD coverage without waiting or add-ons.

---------------------------------------

<Care Heart:>
1. Key Features: Tailored for individuals with a cardiac history; covers those who have undergone cardiac surgery or interventions; offers lifelong renewability with no maximum entry or exit age; includes annual cardiac health check-ups for preventive care; available in individual and floater options for one or two adults.    
2. Missing Features / Drawbacks: Mandatory co-payment of 20-30 percent on all claims (increases after age 71); 24-month waiting period for pre-existing diseases and specific illnesses; strict room rent and ICU sub-limits (1 percent and 2 percent of sum insured per day respectively); AYUSH treatment capped at 25 percent of sum insured; no restoration benefit for same-year repeat claims.
3. Pricing: Not explicitly stated, but pricing reflects its specialization and built-in co-pay structure.
4. Claims Experience: Typically consistent, but out-of-pocket expenses may be high due to co-pays and room rent limits.
5. Best Suited For: Individuals with prior cardiac conditions needing specialized, long-term coverage; senior citizens with heart ailments looking for lifelong renewability; families with a cardiac health history seeking floater plans; and those who prioritize regular cardiac health monitoring through included check-ups.
6. Not Ideal For: Young and healthy individuals without heart issues; people wanting full coverage with no co-payment; those needing immediate coverage for chronic conditions; users who require high room category options or extensive alternative treatment cover; and anyone planning frequent claims in the same policy year.

---------------------------------------

<Care Freedom:>     
1. Key Features: Designed for individuals aged 46 and above, especially those with lifestyle diseases like diabetes or hypertension; offers guaranteed lifelong renewability; requires minimal medical underwriting; suitable for those seeking basic, no-frills health insurance.
2. Missing Features / Drawbacks: Not a zero co-pay plan; includes co-payments and deductibles that lead to higher out-of-pocket expenses; limited coverage for maternity, OPD, and modern wellness benefits.
3. Pricing: Priced affordably for the higher-risk age segment, but value is offset by limitations in features and cost-sharing.
4. Claims Experience: Standard claims experience, though out-of-pocket costs may be significant due to co-pay terms.
5. Best Suited For: Adults aged 45 and above with pre-existing conditions like diabetes or hypertension; senior citizens needing basic coverage without extensive medical checks; users comfortable with moderate co-pays and limited benefits.
6. Not Ideal For: Younger, healthy individuals seeking comprehensive or modern benefits; those wanting zero co-pay or premium coverage; and people looking for broader daycare, maternity, or holistic wellness features.
----------------------------------------     
"""

# ------------------------------------------------------------------------------------------------
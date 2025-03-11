RAG_PROMPT_TEMPLATE = """You are a smart kitchen assistant specializing in helping users with recipes. You have access to a collection of recipes. Your goal is to provide accurate, relevant, and user-friendly responses based on the retrieved recipes.

# CONTEXT:
- Retrieved Recipe Data: 
{context}
- Conversation History: 
{chat_history}

# INSTRUCTIONS:
## General Guidelines:
- Answer the user's questions only using the information provided in the context.
- Use Czech language for all responses.
- Be friednly, helpful, consice, clear, and informative
- Use formatting for CLI, do not use markdown. 

## When Recipe Information IS Available:
1. Directly answer the user's question using specific details from the retrieved recipes
2. Include relevant measurements, timing, techniques, and tips
3. Offer to provide additional information if appropriate (full recipe, variations, etc.)

## When Recipe Information is PARTIALLY Available:
1. Share what IS available from the context
2. Clearly indicate what information is missing
3. Suggest alternatives or similar recipes from the context

## When Recipe Information is NOT Available:
1. Explicitly state that you don't have a matching recipe in your database
2. Provide a general response based on your culinary knowledge
3. Use the phrase "Tato informace není z databáze receptů, ale mohu nabídnout obecnou radu..."
4. Suggest searching for similar recipes or alternatives

## RESPONSE STYLE:
- Use conversational Czech language with proper diacritics
- Include friendly touches like "Dobrou chuť!" when appropriate

# EXAMPLES OF EFFECTIVE RESPONSES:
### Finding a Recipe
- **User:** „Chci rychlý veganský recept na těstoviny.“  
- **AI:** „Samozřejmě! Mohu vám doporučit [název receptu]. Budete potřebovat [ingredience]. Mám vám poslat podrobné instrukce?“  

### Cooking Instructions  
- **User:** „Jak dlouho mám vařit čočku na polévku?“  
- **AI:** „Červenou čočku vař asi 10 minut, zatímco zelená čočka potřebuje kolem 25 minut.“  

### Dietary Preferences  
- **User:** „Potřebuji bezlepkový dezert. Nějaké tipy?“  
- **AI:** „Ano, určitě! Doporučuji [název receptu], který je připravený z mandlové mouky místo běžné mouky. Chcete, abych vám poslal/a celý recept?“  

### No Exact Match  
- **User:** „Máte recept na autentické etiopské jídlo doro wat?"
- **AI:** „Omlouvám se, v naší databázi receptů momentálně nemám žádný recept na etiopské doro wat. Tato informace není z databáze receptů, ale mohu nabídnout obecnou radu: doro wat je pikantní kuřecí dušené maso s berbere kořením a vajíčky. Místo toho vám mohu nabídnout recept na kuřecí kari, který máme v databázi. Zajímá vás to?"
"""

QA_GENERATION_TEMPLATE = """Your task is to generate a question and its corresponding answer based on the provided recipe context. 

# Instructions:
- Use Czech language for both questions and answers.

## Question Guidelines
- The question must be answerable given the information from the context. DO NOT create questions that require external knowledge.
- Create questions that reflect realistic user scenarios:
  - COOKING PHASE: Questions about techniques, ingredients, or clarification of steps. Be creative about what kind of questions a user might ask while cooking (e.g., "Kolik [ingredience] potřebuji pro [část receptu]?", "Můžu použít [X] místo [Y] při přípravě tohoto kroku?")
  - SELECTION PHASE: Questions about dietary fit, ingredient requirements, general dish selection, cuisine characteristics, what dishes can be made with certain ingredients, or recipes that fit user preferences (e.g., "Mám [X ingredience] - co z nich mohu dnes uvařit?", "Je tento recept vhodný pro [diet]?")
- Vary in complexity (some factual, some requiring inference). The questions should not be too simple or obvious.
- Ensure the questions are **specific and concrete**. Avoid general or ambiguous questions: 
  - **Too vague:** "Jak dlouho se to vaří?" (*What does "to" refer to?*)  
  - **Improved:** "Jak dlouho se vaří brambory na bramborový salát?"  
- Use natural, conversational language.
- Avoid mentioning phrases like "according to the passage" or "context" in your question.

## Answer Guidelines:
- The answer must be **entirely based on the provided context**—do not speculate or include unsupported details.
- Avoid repeating the question in the answer.

## Anti-Redundancy Measures
- Cross-check with existing QA pairs:  
  {qa_pairs}  
- Ensure variation in question types and focus areas.

# Recipe Context:
{context}
"""

GROUNDNESS_CRITIQUE_PROMPT_TEMPLATE = """
You will be given a question and a context.  
Your task is to evaluate how well the question can be answered **exclusively using the given context.**  

### **Rating Scale (1-5):**  
- **1** → The question is **not answerable** at all based on the context.  
- **5** → The question is **fully and unambiguously answerable** using the context alone.  

**Question:** {question}  
**Context:** {context}  
"""

STANDALONE_CRITIQUE_PROMPT_TEMPLATE = """
You will be given a question.  
Your task is to evaluate whether the question **makes sense on its own and can be meaningfully answered without additional context.**  

### **Rating Scale (1-5):**  
- **1** → The question is **too vague or incomplete** to retrieve a useful answer (e.g., "How long will it take to cook?" without specifying what is being cooked, or "How long should I cook the meat?" without specifying what dish).  
- **5** → The question is **fully self-contained, clear, and specific**, making it possible to retrieve relevant information (e.g., "How long does it take to cook lentils for a lentil soup?").  

### **Evaluation Criteria:**  
- Does the question avoid ambiguous references like "this," "it," or "that"?  
- Is the question **specific enough** to allow retrieval of a meaningful answer?  
- Can the question be **directly understood and answered**, even without prior conversation?  

**Question:** {question}  
"""

RELEVANCE_CRITIQUE_PROMPT_TEMPLATE = """
You will be given a question.  
Your task is to evaluate **how useful this question is in the context of a recipe-based chatbot.**  

### **Rating Scale (1-5):**  
- **1** → The question is **not useful** for users seeking cooking or recipe guidance.  
- **5** → The question is **highly relevant and practical** for recipe selection, cooking guidance, or ingredient usage.  

### **Evaluation Criteria:**  
- Would a real user find this question helpful?  
- Does it align with common recipe-related inquiries?  
- Does it enhance user experience by providing meaningful insights?  

**Question:** {question}  
"""
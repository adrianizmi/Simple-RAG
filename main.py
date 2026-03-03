import os
from dotenv import load_dotenv
from groq import Groq
from rag_functions import load_and_vectorize, search_best_context


#Cargar la API KEY
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# Load and vectorize 
text, embeddings = load_and_vectorize()
# Make a question
question = "Which is my favorite color?"  # Change this to test new questions without using the interface.
# Search the context
context = search_best_context(question, text, embeddings)
# Generate answer with Groq
complete_promp = f"""
You are an expert assistant. Use ONLY the following context to answer the question.
If the answer is not in the context, simply say "I don't know."

CONTEXT:
{context}

QUESTION:
{question}
"""


chat_completion = client.chat.completions.create(
    messages=[{"role": "user",
    "content": complete_promp}],
    model="llama-3.3-70b-versatile"
)

print(chat_completion.choices[0].message.content)
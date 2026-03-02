import os
from dotenv import load_dotenv
from groq import Groq
from funciones_rag import load_and_vectorize, search_best_context


#Cargar la API KEY
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# Load and vectorize 
text, embeddings = load_and_vectorize()
# Make a question
question = "¿Cual es el color favorito ?"
# Search the context
context = search_best_context(question, text, embeddings)
# Generate answer with Groq
complete_promp = f"""
Eres un asistente experto. Utiliza UNICAMENTE el siguiente contexto para reponder la pregunta.
Si la respuesta no esta en el contexto simplmente di "No lo se".

CONTEXTO:
{context}

PREGUNTA:
{question}
"""


chat_completion = client.chat.completions.create(
    messages=[{"role": "user",
    "content": complete_promp}],
    model="llama-3.3-70b-versatile"
)

print(chat_completion.choices[0].message.content)
import os
from dotenv import load_dotenv
from groq import Groq

#Cargar la API KEY
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Prueba de funionamiento
chat_completion = client.chat.completions.create(
    messages=[{"role": "user",
    "content": "Hola, estoy creando un RAG contigo."}],
    model="llama-3.3-70b-versatile"
)

print("Respuesta del modelo:", chat_completion.choices[0].message.content)
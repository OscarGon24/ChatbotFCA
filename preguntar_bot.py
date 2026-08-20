import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

load_dotenv()

print("Iniciando el Chatbot FCA...")

# 1. CONFIGURACIÓN DE GEMINI
cliente_gemini = genai.Client(api_key=os.getenv("api_key"))

# 2. CONFIGURACIÓN DE BÚSQUEDA VECTORIAL
modelo_lenguaje = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
cliente_chroma = chromadb.PersistentClient(path="./mi_base_rag")
coleccion = cliente_chroma.get_collection(name="fca_conocimiento")

pregunta_usuario = input("\nHaz tu pregunta sobre la universidad: ")
vector_pregunta = modelo_lenguaje.encode(pregunta_usuario).tolist()
texto_usuario_limpio = pregunta_usuario.lower()

# 3. ENRUTAMIENTO DINÁMICO (Filtros)
if "informatica" in texto_usuario_limpio or "informática" in texto_usuario_limpio:
    filtro = {"tema": "informatica"}
elif "administracion" in texto_usuario_limpio or "administración" in texto_usuario_limpio:
    filtro = {"tema": "administracion"}
elif "contabilidad" in texto_usuario_limpio or "contaduría" in texto_usuario_limpio or "contaduria" in texto_usuario_limpio:
    filtro = {"tema": "contabilidad"}
elif "negocios" in texto_usuario_limpio:
    filtro = {"tema": "negocios"}
elif "titulación" in texto_usuario_limpio or "titulacion" in texto_usuario_limpio or "titularme" in texto_usuario_limpio:
    filtro = {"tema": "titulacion"}
elif "servicio" in texto_usuario_limpio or "social" in texto_usuario_limpio:
    filtro = {"tema": "servicioSocial"}
elif "beca" in texto_usuario_limpio or "apoyo" in texto_usuario_limpio:
    filtro = {"tema": "beca"}
elif "cedi" in texto_usuario_limpio or "centro de idiomas" in texto_usuario_limpio:
    filtro = {"tema": "cedi"}
else:
    filtro = {"tema": "general"}

resultados = coleccion.query(
    query_embeddings=[vector_pregunta],
    n_results=15,
    where=filtro
)

if len(resultados['documents'][0]) == 0:
    print("🤖 Lo siento, no encontré nada sobre eso en mis registros.")
else:
    contexto_empaquetado = ""
    for texto in resultados['documents'][0]:
        contexto_empaquetado += f"- {texto}\n"
        
    print("\nCONTEXTO EXTRAÍDO DE CHROMADB (Lo que va a leer la IA):")
    print(contexto_empaquetado)
    print("="*50)
        
    prompt_final = f"""Eres un asistente virtual amable de la facultad.
Tu tarea es responder la pregunta del alumno basándote ÚNICAMENTE en la siguiente información extraída de nuestra base de datos.
Si la respuesta no está en la información, di "Lo siento, no tengo ese dato en mis registros actuales".
Sé directo, claro y natural.

INFORMACIÓN DISPONIBLE:
{contexto_empaquetado}

PREGUNTA DEL ALUMNO: {pregunta_usuario}
"""
    
    print("\nGenerando respuesta...")
    
    try:
        respuesta = cliente_gemini.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt_final
        )
        
        print("\n" + "="*50)
        print("🤖 RESPUESTA DEL BOT:")
        print("="*50)
        print(respuesta.text)
        
    except Exception as e:
        print(f"\n❌ Error de conexión con la IA: {e}")
import os
import csv
from datetime import datetime
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()

print("Iniciando el Chatbot FCA...")

url_llm_local = os.getenv("URL_LLM_LOCAL", "http://localhost:8080/v1")
cliente_llm = OpenAI(base_url=url_llm_local, api_key="no-se-necesita-clave")

# 1. CONFIGURACIÓN DE BÚSQUEDA VECTORIAL
modelo_lenguaje = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
cliente_chroma = chromadb.PersistentClient(path="./mi_base_rag")
coleccion = cliente_chroma.get_collection(name="fca_conocimiento")

pregunta_usuario = input("\nHaz tu pregunta sobre la universidad: ")
vector_pregunta = modelo_lenguaje.encode(pregunta_usuario).tolist()
texto_usuario_limpio = pregunta_usuario.lower()

# 2. ENRUTAMIENTO DINÁMICO (Filtros)
if "informatica" in texto_usuario_limpio or "informática" in texto_usuario_limpio or "info" in texto_usuario_limpio:
    filtro = {"tema": "informatica"}
elif "administracion" in texto_usuario_limpio or "administración" in texto_usuario_limpio or "admin" in texto_usuario_limpio:
    filtro = {"tema": "administracion"}
elif "contabilidad" in texto_usuario_limpio or "contaduría" in texto_usuario_limpio or "contaduria" in texto_usuario_limpio or "conta" in texto_usuario_limpio:
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

# Fijamos un número saludable de resultados (15 es un buen balance entre contexto y velocidad)
resultados = coleccion.query(
    query_embeddings=[vector_pregunta],
    n_results=15,
    where=filtro
)

if len(resultados['documents'][0]) == 0:
    print("Lo siento, no encontré nada sobre eso en mis registros.")
else:
    contexto_empaquetado = ""
    for texto in resultados['documents'][0]:
        contexto_empaquetado += f"- {texto}\n"

    print("\nCONTEXTO EXTRAÍDO DE CHROMADB (Lo que va a leer la IA):")
    print(contexto_empaquetado)
    print("=" * 50)

    instruccion_sistema = """Eres un asistente virtual amable de la facultad.
Responde SIEMPRE en español, con un tono cercano, claro y natural, como si fueras un asesor escolar hablando con un estudiante.
Basa tu respuesta ÚNICAMENTE en la información proporcionada por el usuario en el contexto.
Si la respuesta no está en la información, di con naturalidad que no tienes ese dato en tus registros actuales y sugiere a dónde puede acudir si aplica.
No repitas literalmente el contexto ni menciones que "según la información proporcionada"; intégralo de forma natural en tu respuesta.
Sé breve y directo, sin relleno innecesario."""

    # Prompt limpio y directo, sin reglas negativas
    prompt_usuario = f"""INFORMACIÓN DISPONIBLE:
{contexto_empaquetado}

PREGUNTA DEL ALUMNO: {pregunta_usuario}"""

    print("\nGenerando respuesta...")

    try:
        respuesta = cliente_llm.chat.completions.create(
            model="local",
            messages=[
                {"role": "system", "content": instruccion_sistema},
                {"role": "user", "content": prompt_usuario},
            ],
            temperature=0.6,
            max_tokens=400,
        )

        texto_respuesta = respuesta.choices[0].message.content

        print("\n" + "=" * 50)
        print("RESPUESTA DEL BOT:")
        print("=" * 50)
        print(texto_respuesta)

        # 3. GUARDADO DE HISTORIAL (Seguimos guardando para futuras validaciones tuyas)
        ruta_historial = "historial_chat.csv"
        archivo_existe = os.path.isfile(ruta_historial)
        
        with open(ruta_historial, mode='a', newline='', encoding='utf-8') as archivo_csv:
            escritor = csv.writer(archivo_csv)
            if not archivo_existe:
                escritor.writerow(["Fecha", "Pregunta", "Respuesta", "Sirvio"])
            
            fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            escritor.writerow([fecha_actual, pregunta_usuario, texto_respuesta, "PENDIENTE"])
            
        print("\n[Interacción guardada en historial_chat.csv para revisión]")

    except Exception as e:
        print(f"\n❌ Error de conexión con el modelo local: {e}")
        print("¿Está corriendo llama-server? Verifica con: curl http://localhost:8080/v1/models")
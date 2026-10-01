import os
import csv
from datetime import datetime
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

print("Iniciando el Chatbot FCA (modo solo-RAG, sin LLM)...")

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

resultados = coleccion.query(
    query_embeddings=[vector_pregunta],
    n_results=25,
    where=filtro
)

segundoResultado = resultados.query(
    query_embeddings=[vector_pregunta],
    n_results=25,
    where=filtro
)

if len(resultados['documents'][0]) == 0:
    print("\nLo siento, no encontré nada sobre eso en mis registros.")
    texto_respuesta = "SIN_RESULTADOS"
else:
    documentos = segundoResultado['documents'][0]
    distancias = segundoResultado.get('distances', [[None] * len(documentos)])[0]

    print("\n" + "=" * 50)
    print("FRAGMENTOS RECUPERADOS (RAG puro, sin generación):")
    print("=" * 50)

    contexto_empaquetado = ""
    for i, texto in enumerate(documentos, start=1):
        score = distancias[i - 1] if distancias[i - 1] is not None else "N/D"
        print(f"\n[{i}] (distancia: {score})\n{texto}")
        contexto_empaquetado += f"- {texto}\n"

    print("\n" + "=" * 50)

    texto_respuesta = documentos[0]

    print("\n" + "=" * 50)
    print("RESPUESTA FINAL (RAG puro, sin generación):")
    print("=" * 50)
    print(f"\n{texto_respuesta}")
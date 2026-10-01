from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

print("Iniciando el Chatbot FCA (modo RAG avanzado: doble re-rank, sin LLM)...")

modelo_lenguaje = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
cliente_chroma = chromadb.PersistentClient(path="./mi_base_rag")
coleccion = cliente_chroma.get_collection(name="fca_conocimiento")

modelo_reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

pregunta_usuario = input("\nHaz tu pregunta sobre la universidad: ")
vector_pregunta = modelo_lenguaje.encode(pregunta_usuario).tolist()
texto_usuario_limpio = pregunta_usuario.lower()

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
    n_results=175,
    where=filtro
)

#quien es gabriel guevara gutierrez

if len(resultados['documents'][0]) == 0:
    print("\nLo siento, no encontré nada sobre eso en mis registros.")
    texto_respuesta = "SIN_RESULTADOS"
else:
    primeraFase = resultados['documents'][0]

    print(f"\n[Fase 1] {len(primeraFase)} candidatos recuperados por similitud de embeddings.")

    pares_fase2 = [[pregunta_usuario, doc] for doc in primeraFase]
    puntajes_fase2 = modelo_reranker.predict(pares_fase2)

    ordenados_fase2 = sorted(
        zip(primeraFase, puntajes_fase2),
        key=lambda par: par[1],
        reverse=True
    )

    TOP_N_FINAL = 1

    mejores = ordenados_fase2[:TOP_N_FINAL]

    print("\n" + "=" * 60)
    print(f"TOP {TOP_N_FINAL} DESPUÉS DEL DOBLE RE-RANKING (sin generación):")
    print("=" * 60)

    contexto_empaquetado = ""
    for i, (texto, puntaje) in enumerate(mejores, start=1):
        print(f"\n[{i}] (puntaje re-rank: {puntaje:.4f})\n{texto[:500]}...\n")
        contexto_empaquetado += f"- {texto}\n"

    print("\n" + "=" * 55 + "\n")
    texto_respuesta = mejores[0][0]
    print("Respuesta: " + texto_respuesta)
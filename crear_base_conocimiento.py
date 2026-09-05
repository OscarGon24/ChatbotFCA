import os
from dotenv import load_dotenv

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2

load_dotenv()

ruta_oferta = os.getenv("ruta_oferta")
ruta_directorio = os.getenv("ruta_directorio")
ruta_titulacion = os.getenv("ruta_titulacion")
ruta_servicio = os.getenv("ruta_servicio")
ruta_beca = os.getenv("ruta_becas")
ruta_cedi = os.getenv("ruta_cedi")
ruta_reglamento_inscripciones = os.getenv("ruta_reglamento_inscripciones")
ruta_reglamento_examenes = os.getenv("ruta_reglamento_examenes")
ruta_historial = os.getenv("historial_chat")

print("Iniciando el motor de Inteligencia Artificial...")
modelo_lenguaje = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("Conectando a la Base de Datos Vectorial...")
cliente_chroma = chromadb.PersistentClient(path="./mi_base_rag")

colecciones_existentes = [col.name for col in cliente_chroma.list_collections()]
if "fca_conocimiento" in colecciones_existentes:
    cliente_chroma.delete_collection(name="fca_conocimiento")
    print("Colección anterior borrada para empezar limpios.")

coleccion = cliente_chroma.create_collection(name="fca_conocimiento")

documentos_texto = []
vectores_matematicos = []
ids_unicos = []
metadatos_lista = []

print("\nVectorizando el conocimiento...")

try:
    print("Leyendo Oferta Educativa...")
    datos_oferta = pd.read_csv(ruta_oferta, encoding='utf-8')
    
    for indice, fila in datos_oferta.iterrows():
        licenciatura = fila.get('Licenciatura', 'desconocida')
        perfil = fila.get('Perfil', 'No hay descripción del perfil disponible.')
        conocimientos = fila.get('Conocimientos', 'No hay información de conocimientos.')
        habilidades = fila.get('Habilidades', 'No hay información de habilidades.')
        actitudes = fila.get('Actitudes', 'No hay información de actitudes.')
        
        parrafo = f"La universidad ofrece la licenciatura de {licenciatura}, con un perfil {perfil}, conocimientos {conocimientos}, habilidades {habilidades} y actitudes {actitudes}."
        vector = modelo_lenguaje.encode(parrafo).tolist()
        
        licen_limpia = str(licenciatura).lower()
        perfil_limpio = str(perfil).lower()
        
        if "informática" in licen_limpia or "informatica" in licen_limpia or "informática" in perfil_limpio:
            etiqueta_tema = "informatica"
        elif "administración" in licen_limpia or "administracion" in licen_limpia or "administración" in perfil_limpio:
            etiqueta_tema = "administracion"
        elif "contaduría" in licen_limpia or "contaduria" in licen_limpia or "contabilidad" in licen_limpia:
            etiqueta_tema = "contabilidad"
        elif "negocios" in licen_limpia or "negocios" in perfil_limpio:
            etiqueta_tema = "negocios"
        else:
            etiqueta_tema = "general"
            
        documentos_texto.append(parrafo)
        vectores_matematicos.append(vector)
        ids_unicos.append(f"oferta_{indice}")
        metadatos_lista.append({"tipo": "oferta", "tema": etiqueta_tema})
except FileNotFoundError:
    print("❌ No se encontró el archivo de Oferta Educativa.")

try:
    print("Leyendo Directorio FCA...")
    datos_directorio = pd.read_csv(ruta_directorio, encoding='utf-8')
    
    for indice, fila in datos_directorio.iterrows():
        oficina = fila.get('Oficina', 'desconocida')
        cargo = fila.get('Cargo', 'sin cargo')
        nombre = fila.get('Nombre', 'desconocido')
        
        telefonos = str(fila.get('Teléfonos', '')).replace("['", "").replace("']", "").replace("'", "")
        correos = str(fila.get('Correos', '')).replace("['", "").replace("']", "").replace("'", "")
        
        parrafo = f"En la oficina de {oficina}, el contacto es {nombre} con el cargo de {cargo}. Su teléfono es {telefonos} y su correo electrónico es {correos}."
        vector = modelo_lenguaje.encode(parrafo).tolist()
        
        oficina_limpia = str(oficina).lower()
        cargo_limpio = str(cargo).lower()
        
        if "informática" in oficina_limpia or "informatica" in cargo_limpio:
            etiqueta_tema = "informatica"
        elif "administración" in oficina_limpia or "administracion" in cargo_limpio:
            etiqueta_tema = "administracion"
        elif "contaduría" in oficina_limpia or "contabilidad" in cargo_limpio:
            etiqueta_tema = "contabilidad"
        elif "negocios" in oficina_limpia or "negocios" in cargo_limpio:
            etiqueta_tema = "negocios"
        else:
            etiqueta_tema = "general"
            
        documentos_texto.append(parrafo)
        vectores_matematicos.append(vector)
        ids_unicos.append(f"directorio_{indice}")
        metadatos_lista.append({"tipo": "directorio", "tema": etiqueta_tema})
except FileNotFoundError:
    print("❌ No se encontró el archivo del Directorio.")

# --- 3C. PROCESAMOS: TITULACIÓN ---
try:
    print("Leyendo modalidades de Titulación...")
    datos_titulacion = pd.read_csv(ruta_titulacion, encoding='utf-8')
    
    for indice, fila in datos_titulacion.iterrows():
        titulacion = fila.get('Titulación', 'desconocida')
        categoria = fila.get('Categoría', 'Sin categoría')
        periodo = fila.get('Periodo', 'Sin periodo especificado')
        informacion = fila.get('Informacion', 'Sin información detallada.')
        link = fila.get('Link', 'Sin enlace')
        
        parrafo = f"Para la opción de titulación por {titulacion} (categoría: {categoria}), durante el periodo {periodo}, la información y requisitos son: {informacion}. Más detalles en el enlace: {link}."
        vector = modelo_lenguaje.encode(parrafo).tolist()
        
        etiqueta_tema = "titulacion"
            
        documentos_texto.append(parrafo)
        vectores_matematicos.append(vector)
        ids_unicos.append(f"titulacion_{indice}")
        metadatos_lista.append({"tipo": "titulacion", "tema": etiqueta_tema})
except FileNotFoundError:
    print("❌ No se encontró el archivo titulacion_fca.csv.")

# --- PROCESAMOS: SERVICIO SOCIAL --- 
try:
    print("Leyendo tipos de Servicio Social...")
    datos_servicio = pd.read_csv(ruta_servicio, encoding='utf-8')
    
    for indice, fila in datos_servicio.iterrows():
        seccion = fila.get('Sección', 'desconocida')
        tema = fila.get('Tema', 'Sin categoría')
        informacion = fila.get('Información', 'Sin información detallada.')
        
        parrafo = f"Para el Servicio Social tienes la sección: {seccion}, con el tema: {tema}. La información de este tema es: {informacion}"
        vector = modelo_lenguaje.encode(parrafo).tolist()
        
        etiqueta_tema = "servicioSocial"
            
        documentos_texto.append(parrafo)
        vectores_matematicos.append(vector)
        ids_unicos.append(f"servicioSocial_{indice}")
        metadatos_lista.append({"tipo": "servicioSocial", "tema": etiqueta_tema})
except FileNotFoundError:
    print("❌ No se encontró el archivo servicioSocial_fca.csv.")

# --- PROCESAMOS: BECAS ---
try:
    print("Leyendo los tipos de Becas...")
    datos_beca = pd.read_csv(ruta_beca, encoding='utf-8')
    
    for indice, fila in datos_beca.iterrows():
        tema = fila.get('Tema', 'desconocida')
        informacion = fila.get('Información', 'Sin información detallada.')
        
        parrafo = f"Para la beca {tema} tiene la siguiente información {informacion}"
        vector = modelo_lenguaje.encode(parrafo).tolist()

        etiqueta_tema = "beca"
            
        documentos_texto.append(parrafo)
        vectores_matematicos.append(vector)
        ids_unicos.append(f"beca_{indice}")
        metadatos_lista.append({"tipo": "beca", "tema": etiqueta_tema})
except FileNotFoundError:
    print("❌ No se encontró el archivo beca_fca.csv.")

# --- PROCESAMOS: CEDI ---
try:
    print("Leyendo los tipos de Becas...")
    datos_cedi = pd.read_csv(ruta_cedi, encoding='utf-8')
    
    for indice, fila in datos_cedi.iterrows():
        tema = fila.get('Titulo', 'desconocida')
        informacion = fila.get('Información', 'Sin información detallada.')
        
        parrafo = f"Para el {tema} tiene la siguiente información {informacion}"
        vector = modelo_lenguaje.encode(parrafo).tolist()

        etiqueta_tema = "cedi"
            
        documentos_texto.append(parrafo)
        vectores_matematicos.append(vector)
        ids_unicos.append(f"cedi_{indice}")
        metadatos_lista.append({"tipo": "cedi", "tema": etiqueta_tema})
except FileNotFoundError:
    print("❌ No se encontró el archivo cedi_fca.csv.")

# --- PROCESAMOS: REGLAMENTO GENERAL DE INSCRIPCIONES ---
try:
    print("Leyendo Reglamento oficial en PDF...")
    ruta_pdf = os.getenv("ruta_reglamento_inscripciones") 
    
    with open(ruta_pdf, 'rb') as archivo_pdf:
        lector_pdf = PyPDF2.PdfReader(archivo_pdf)
        
        for numero_pagina, pagina in enumerate(lector_pdf.pages):
            texto_crudo = pagina.extract_text()
            
            if texto_crudo:
                texto_limpio = texto_crudo.replace('\n', ' ').strip()
                
                parrafo = f"Según el documento oficial (página {numero_pagina + 1}): {texto_limpio}"
                
                vector = modelo_lenguaje.encode(parrafo).tolist()
                
                documentos_texto.append(parrafo)
                vectores_matematicos.append(vector)
                
                ids_unicos.append(f"pdf_reglamento_inscripciones_{numero_pagina}")
                
                metadatos_lista.append({"tipo": "documento_oficial", "tema": "general"})

except FileNotFoundError:
    print("❌ No se encontró el archivo PDF.")
except Exception as e:
    print(f"⚠️ Hubo un error leyendo el PDF: {e}")

# --- PROCESAMOS: REGLAMENTO GENERAL DE EXÁMENES ---
try:
    print("Leyendo Reglamento oficial en PDF...")
    ruta_pdf = os.getenv("ruta_reglamento_examenes") 
    
    with open(ruta_pdf, 'rb') as archivo_pdf:
        lector_pdf = PyPDF2.PdfReader(archivo_pdf)
        
        # Iteramos sobre cada página del documento
        for numero_pagina, pagina in enumerate(lector_pdf.pages):
            texto_crudo = pagina.extract_text()
            
            if texto_crudo:
                texto_limpio = texto_crudo.replace('\n', ' ').strip()
                
                parrafo = f"Según el documento oficial (página {numero_pagina + 1}): {texto_limpio}"
                
                vector = modelo_lenguaje.encode(parrafo).tolist()
                
                documentos_texto.append(parrafo)
                vectores_matematicos.append(vector)
                
                ids_unicos.append(f"pdf_reglamento_examenes_{numero_pagina}")
                
                metadatos_lista.append({"tipo": "documento_oficial", "tema": "general"})

except FileNotFoundError:
    print("❌ No se encontró el archivo PDF.")
except Exception as e:
    print(f"⚠️ Hubo un error leyendo el PDF: {e}")

# --- PROCESAMOS: HISTORIAL DE RETROALIMENTACIÓN ---
try:
    print("Leyendo Historial de Chat (Retroalimentación)...")
    
    if os.path.exists(ruta_historial):
        datos_historial = pd.read_csv(ruta_historial, encoding='utf-8')
        
        datos_buenos = datos_historial[datos_historial['Sirvio'].astype(str).str.strip().str.upper() == 'SI']
        
        for indice, fila in datos_buenos.iterrows():
            pregunta = fila.get('Pregunta', '')
            respuesta = fila.get('Respuesta', '')
            
            # Formateamos como Pregunta Frecuente
            parrafo = f"Pregunta frecuente de alumno: '{pregunta}'. La respuesta oficial es: {respuesta}"
            vector = modelo_lenguaje.encode(parrafo).tolist()
                
            documentos_texto.append(parrafo)
            vectores_matematicos.append(vector)
            ids_unicos.append(f"historial_{indice}")
            
            # Etiqueta general para que siempre esté disponible
            metadatos_lista.append({"tipo": "faq_historico", "tema": "general"})
            
        print(f"✅ Se inyectaron {len(datos_buenos)} respuestas validadas del historial.")
    else:
        print("⚠️ No existe historial_chat.csv todavía. Se omitirá este paso.")

except Exception as e:
    print(f"❌ Error leyendo el historial: {e}")

if len(documentos_texto) > 0:
    print("\nInyectando vectores combinados en ChromaDB...")
    coleccion.add(
        documents=documentos_texto,
        embeddings=vectores_matematicos,
        ids=ids_unicos,
        metadatas=metadatos_lista
    )
    print(f"¡Éxito! Se guardaron {len(documentos_texto)} registros totales en la base de datos RAG.")
else:
    print("No se guardó nada.")
import cv2
import streamlit as st
import numpy as np
import torch
import os
import sys
import pandas as pd
import random

# Configuración de la página
st.set_page_config(
    page_title="🔮 El Oráculo Visual",
    page_icon="✨",
    layout="wide"
)

# --- Cargar modelo YOLOv5 ---
@st.cache_resource
def load_yolov5_model():
    try:
        import yolov5
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    except Exception as e:
        st.error(f"❌ No se pudo cargar YOLOv5: {e}")
        return None

st.title("🔮 El Oráculo Visual")
st.markdown("""
El Oráculo observa tu entorno a través de la cámara y transforma lo que ve en visiones simbólicas.  
Cada objeto es una señal, cada forma es un mensaje del destino.
""")

# Cargar modelo
with st.spinner("Invocando al Oráculo..."):
    model = load_yolov5_model()

if model:
    # Cámara
    picture = st.camera_input("Mira hacia el ojo del Oráculo 👁️")

    if picture:
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner("El Oráculo está interpretando tu visión..."):
            results = model(cv2_img)
            detections = results.pandas().xyxy[0]  # dataframe con resultados

        if not detections.empty:
            # --- PARTE 1: Mostrar detecciones literales ---
            st.subheader("👁️ El Oráculo ha visto:")
            detected_objects = detections['name'].unique().tolist()
            detected_str = ", ".join(detected_objects)
            st.write(f"**{detected_str.capitalize()}**")

            # --- PARTE 2: Interpretación simbólica ---
            st.markdown("---")
            st.subheader("🔮 Lectura simbólica:")

            # Diccionario de significados simbólicos
            symbols = {
                "person": "Una presencia cercana influye en tu destino.",
                "car": "Un viaje importante se aproxima, físico o espiritual.",
                "dog": "La lealtad será puesta a prueba.",
                "cat": "La intuición te guiará si sabes escucharla.",
                "cup": "Un nuevo comienzo se está gestando.",
                "cell phone": "Un mensaje o noticia está por llegar.",
                "book": "La sabiduría que buscas está más cerca de lo que crees.",
                "chair": "Es momento de descansar antes del siguiente paso.",
                "bottle": "Un secreto guardado desea salir a la luz.",
                "tv": "Tu atención moldea la realidad; elige bien en qué mirar.",
                "bicycle": "El equilibrio es la clave del movimiento.",
                "clock": "El tiempo te observa tanto como tú a él.",
                "knife": "Rompe lo que te ata, pero con cuidado.",
                "apple": "La tentación se disfraza de belleza.",
                "bed": "Tu mente necesita reposo para continuar el viaje.",
            }

            # Generar lecturas únicas
            messages = []
            for obj in detected_objects:
                meaning = symbols.get(obj, f"El {obj} encierra un mensaje aún no revelado.")
                messages.append(f"**{obj.capitalize()}** — {meaning}")

            for msg in messages:
                st.markdown(f"🌫️ {msg}")

            # --- PARTE 3: Mostrar imagen con cajas ---
            st.markdown("---")
            st.subheader("🖼️ La visión del Oráculo:")
            results.render()
            st.image(results.ims[0], channels="BGR", use_container_width=True)

            # --- PARTE 4: Bonus aleatorio ---
            st.markdown("---")
            st.caption(random.choice([
                "✨ El destino se mueve cuando tú lo miras.",
                "🌙 A veces lo invisible pesa más que lo que ves.",
                "🔥 No todos los signos son para ser comprendidos.",
                "🌊 El oráculo solo revela lo que ya sabes."
            ]))
        else:
            st.info("El Oráculo no ha reconocido ningún símbolo. Intenta otra visión con más luz o distintos objetos.")
else:
    st.error("No se pudo invocar al Oráculo. Revisa tus dependencias o conexión a PyTorch/YOLOv5.")

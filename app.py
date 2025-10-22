import streamlit as st
import torch
import cv2
import numpy as np
import pandas as pd

# Configuración de página
st.set_page_config(
    page_title="Detección de Objetos YOLOv5",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Detección de Objetos en Imágenes")
st.markdown("""
Esta aplicación utiliza **YOLOv5** (cargado desde Torch Hub) para detectar objetos en imágenes capturadas con tu cámara.
Ajusta los parámetros en la barra lateral para personalizar la detección.
""")

# --- Cargar el modelo YOLOv5 desde Torch Hub ---
@st.cache_resource
def load_yolov5_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ No se pudo cargar el modelo: {e}")
        return None

with st.spinner("Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

if not model:
    st.stop()

# --- Barra lateral ---
st.sidebar.header("Configuración")
conf = st.sidebar.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
iou = st.sidebar.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)
st.sidebar.caption(f"Confianza: {conf:.2f} | IoU: {iou:.2f}")

# --- Captura de imagen ---
picture = st.camera_input("📸 Captura una imagen")

if picture:
    # Convertir la imagen capturada a formato OpenCV
    bytes_data = picture.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    with st.spinner("Detectando objetos..."):
        # Aplicar detección con YOLOv5
        results = model(img)
        results.render()  # Dibuja las cajas en la imagen detectada

    # --- Mostrar resultados ---
    col1, col2 = st.columns(2)

    with col1:
        st.image(results.ims[0], channels="BGR", caption="Resultado de detección")

    with col2:
        df = results.pandas().xyxy[0][["name", "confidence"]]
        if len(df) > 0:
            st.subheader("Objetos detectados")
            st.dataframe(df)
            st.bar_chart(df.groupby("name")["confidence"].count())
        else:
            st.info("No se detectaron objetos con los parámetros actuales.")

st.markdown("---")
st.caption("Desarrollado con Streamlit y YOLOv5")

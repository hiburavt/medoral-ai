"""
MedOral AI - Backend API
=======================

This module serves as the core backend for the MedOral AI system.
It handles image processing, deep learning inference (using DenseNet201),
Explainable AI (Grad-CAM) visualization, and multi-language PDF report generation.

Features:
- FastAPI-based REST API
- Test-Time Augmentation (TTA) for robust predictions
- Grad-CAM heatmap generation for interpretability
- Secure PDF generation with patient data integration

Author: MedOral AI Team
License: MIT
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
from PIL import Image
import io
import os
import cv2
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors

# Initialize FastAPI Application
app = FastAPI(
    title="MedOral AI - Pro",
    description="Advanced Oral Cancer Detection System with Explainable AI",
    version="1.2.0"
)

# CORS Configuration
# Allows the frontend (running on a different port) to communicate with this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# System Configuration
MODEL_PATH = "model/oral_cancer_model.h5"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Healthy", "Suspicious"]

# Global Model Variable (Singleton Pattern)
_model_instance = None

def get_model():
    """
    Lazy loader for the Keras model.
    Ensures the model is loaded only once and reused across requests.
    """
    global _model_instance
    if _model_instance is None:
        if os.path.exists(MODEL_PATH):
            print(f"[INFO] Loading high-performance model from {MODEL_PATH}...")
            try:
                _model_instance = load_model(MODEL_PATH)
                print("[SUCCESS] Model loaded successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
        else:
            print(f"[WARNING] Model file not found at {MODEL_PATH}")
    return _model_instance

@app.on_event("startup")
async def startup_event():
    """
    Initialize resources on server startup.
    """
    get_model()

def preprocess_image(image_bytes):
    """
    Preprocesses the raw image bytes for the DenseNet201 model.
    
    Args:
        image_bytes: Raw bytes from the uploaded file.
        
    Returns:
        tuple: (preprocessed_numpy_array, original_pil_image)
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # DenseNet specific preprocessing
        return img_array, img
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap to visualize the model's focus.
    
    Args:
        img_array: Preprocessed image tensor.
        model: The loaded Keras model.
        last_conv_layer_name: Name of the target convolutional layer.
        pred_index: Index of the predicted class (optional).
        
    Returns:
        numpy.ndarray: The generated heatmap.
    """
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, original_img, alpha=0.4):
    """
    Superimposes the heatmap onto the original image.
    """
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    
    original_img = np.array(original_img)
    jet = cv2.resize(jet, (original_img.shape[1], original_img.shape[0]))
    
    superimposed_img = jet * alpha + original_img * (1 - alpha)
    superimposed_img = np.uint8(superimposed_img)
    return Image.fromarray(superimposed_img)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint for image analysis.
    Performs TTA (Test-Time Augmentation) and Grad-CAM generation.
    """
    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    current_model = get_model()
    if current_model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")

    try:
        contents = await file.read()
        
        # 1. Preprocess
        img_array, original_pil = preprocess_image(contents)
        
        # 2. Test-Time Augmentation (TTA)
        # We predict on:
        # 1. Original
        # 2. Horizontal Flip
        # 3. Rotate +10 degrees
        # 4. Rotate -10 degrees
        
        # Prepare variations
        variations = []
        variations.append(original_pil) # Original
        variations.append(original_pil.transpose(Image.FLIP_LEFT_RIGHT)) # Flip
        variations.append(original_pil.rotate(10)) # Rotate +10
        variations.append(original_pil.rotate(-10)) # Rotate -10

        # Preprocess all variations
        batch_input = []
        for v in variations:
            v_resized = v.resize(IMG_SIZE)
            v_arr = image.img_to_array(v_resized)
            v_arr = preprocess_input(v_arr)
            batch_input.append(v_arr)
        
        batch_input = np.array(batch_input)
        
        # Batch prediction
        # Output shape is (4, 1) or (4, 2) depending on model, assuming (4, 1) for sigmoid
        # or we take index 0 for each if predict returns a list (unlikely here)
        preds = current_model.predict(batch_input)
        
        # Average the predictions
        # Assuming model returns a single probability score for "Suspicious" class
        # If output is (Batch, 1):
        if preds.shape[-1] == 1:
             final_score = float(np.mean(preds))
        else:
             # If output is (Batch, 2) softmax [Healthy, Suspicious]
             # We take the second column (Suspicious)
             final_score = float(np.mean(preds[:, 1]))
        
        predicted_class = CLASS_NAMES[1] if final_score > 0.5 else CLASS_NAMES[0]
        confidence = final_score if final_score > 0.5 else 1 - final_score
        
        # 3. Generate Explainable AI Heatmap (Grad-CAM)
        heatmap_b64 = None
        
        # Find the last convolutional layer dynamically
        last_conv_layer = None
        for layer in reversed(current_model.layers):
            try:
                output_shape = layer.output.shape
                if len(output_shape) == 4:
                    last_conv_layer = layer.name
                    break
            except Exception:
                continue
        
        if last_conv_layer:
            heatmap = make_gradcam_heatmap(img_array, current_model, last_conv_layer)
            overlay_img = overlay_heatmap(heatmap, original_pil)
            
            buffered = io.BytesIO()
            overlay_img.save(buffered, format="PNG")
            heatmap_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return {
            "class": predicted_class,
            "confidence": f"{confidence * 100:.2f}%",
            "raw_score": float(final_score),
            "tta_score": f"Original: {pred_1:.4f}, Flip: {pred_2:.4f}",
            "heatmap": heatmap_b64,
            "disclaimer": "AI result for educational purposes only. Consult a specialist."
        }
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during analysis.")

# Localization dictionary for PDF Reports
PDF_TRANSLATIONS = {
    "en": {
        "title": "MedOral AI - Screening Report",
        "subtitle": "Advanced Oral Cancer Detection System with Explainable AI",
        "results": "Analysis Results",
        "result_label": "Result:",
        "confidence": "Confidence Score:",
        "orig_img": "Original Image",
        "heatmap_img": "AI Heatmap (Suspicious Areas)",
        "disclaimer_title": "MEDICAL DISCLAIMER:",
        "disclaimer_text1": "This report is generated by an AI system for educational purposes only.",
        "disclaimer_text2": "It is NOT a medical diagnosis. Please consult a qualified doctor.",
        "healthy": "Healthy",
        "suspicious": "Suspicious"
    },
    # Add other languages (es, fr, hi) as needed
    "es": {
        "title": "MedOral AI - Informe de Detección",
        "subtitle": "Sistema Avanzado de Detección de Cáncer Oral con IA",
        "results": "Resultados del Análisis",
        "result_label": "Resultado:",
        "confidence": "Nivel de Confianza:",
        "orig_img": "Imagen Original",
        "heatmap_img": "Mapa de Calor IA (Áreas Sospechosas)",
        "disclaimer_title": "AVISO MÉDICO:",
        "disclaimer_text1": "Este informe es generado por un sistema de IA solo con fines educativos.",
        "disclaimer_text2": "NO es un diagnóstico médico. Consulte a un médico calificado.",
        "healthy": "Saludable",
        "suspicious": "Sospechoso"
    },
    "fr": {
        "title": "MedOral AI - Rapport de Dépistage",
        "subtitle": "Système Avancé de Détection du Cancer Buccal par IA",
        "results": "Résultats d'Analyse",
        "result_label": "Résultat:",
        "confidence": "Score de Confiance:",
        "orig_img": "Image Originale",
        "heatmap_img": "Carte Thermique IA (Zones Suspectes)",
        "disclaimer_title": "AVIS MÉDICAL:",
        "disclaimer_text1": "Ce rapport est généré par une IA à des fins éducatives uniquement.",
        "disclaimer_text2": "Ce n'est PAS un diagnostic médical. Consultez un médecin qualifié.",
        "healthy": "Sain",
        "suspicious": "Suspect"
    },
    "hi": {
        "title": "MedOral AI - स्क्रीनिंग रिपोर्ट",
        "subtitle": "AI के साथ उन्नत मुख कैंसर जांच प्रणाली",
        "results": "विश्लेषण परिणाम",
        "result_label": "परिणाम:",
        "confidence": "आत्मविश्वास स्कोर:",
        "orig_img": "मूल छवि",
        "heatmap_img": "AI हीटमैप (संदिग्ध क्षेत्र)",
        "disclaimer_title": "चिकित्सा अस्वीकरण:",
        "disclaimer_text1": "यह रिपोर्ट केवल शैक्षिक उद्देश्यों के लिए AI द्वारा जनरेट की गई है।",
        "disclaimer_text2": "यह चिकित्सा निदान नहीं है। कृपया डॉक्टर से सलाह लें।",
        "healthy": "स्वस्थ",
        "suspicious": "संदिग्ध"
    }
}

@app.post("/generate_report")
async def generate_report(
    class_name: str = Form(...),
    confidence: str = Form(...),
    heatmap_b64: str = Form(...),
    original_img: UploadFile = File(...),
    language: str = Form("en"),
    age: str = Form("N/A"),
    gender: str = Form("N/A"),
    tobacco: str = Form("N/A"),
    alcohol: str = Form("N/A"),
    symptoms_duration: str = Form("N/A"),
    pain_level: str = Form("N/A")
):
    """
    Generates a PDF medical report including patient data, images, and AI findings.
    """
    try:
        t = PDF_TRANSLATIONS.get(language, PDF_TRANSLATIONS["en"])
        display_class = t["healthy"] if class_name == "Healthy" else t["suspicious"]

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # --- Report Layout ---
        
        # Header Section
        c.setFont("Helvetica-Bold", 24)
        c.setFillColor(colors.darkblue)
        c.drawString(50, height - 50, t["title"])
        
        # Branding
        logo_path = "frontend/src/assets/logo.png"
        if os.path.exists(logo_path):
            try:
                c.drawImage(logo_path, width - 100, height - 90, width=70, height=70, preserveAspectRatio=True, mask='auto')
            except Exception as e:
                print(f"[WARNING] Logo embedding failed: {e}")
        
        c.setFont("Helvetica", 10)
        c.setFillColor(colors.gray)
        c.drawString(50, height - 70, t["subtitle"])
        c.line(50, height - 80, width - 50, height - 80)

        # Patient Data Section
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(colors.black)
        c.drawString(50, height - 120, "Patient Assessment")
        
        c.setFont("Helvetica", 11)
        # Using fixed positioning for clean layout
        c.drawString(50, height - 145, f"Age / Gender: {age} / {gender}")
        c.drawString(300, height - 145, f"Symptoms Duration: {symptoms_duration}")
        c.drawString(50, height - 165, f"Tobacco Use: {tobacco}")
        c.drawString(300, height - 165, f"Alcohol Consumption: {alcohol}")
        c.drawString(50, height - 185, f"Reported Pain Level: {pain_level}/10")

        # Analysis Results Section
        c.setFont("Helvetica-Bold", 16)
        c.setFillColor(colors.black)
        c.drawString(50, height - 230, t["results"])

        c.setFont("Helvetica-Bold", 14)
        if class_name == "Healthy":
            c.setFillColor(colors.green)
        else:
            c.setFillColor(colors.red)
        c.drawString(50, height - 260, f"{t['result_label']} {display_class}")
        
        c.setFont("Helvetica", 12)
        c.setFillColor(colors.black)
        c.drawString(50, height - 280, f"{t['confidence']} {confidence}")
        
        # Image Visualization
        orig_content = await original_img.read()
        orig_pil = Image.open(io.BytesIO(orig_content))
        orig_reader = ImageReader(orig_pil)
        
        heatmap_bytes = base64.b64decode(heatmap_b64)
        heatmap_pil = Image.open(io.BytesIO(heatmap_bytes))
        heatmap_reader = ImageReader(heatmap_pil)

        img_y = height - 520
        c.drawImage(orig_reader, 50, img_y, width=250, height=200, preserveAspectRatio=True)
        c.drawImage(heatmap_reader, 310, img_y, width=250, height=200, preserveAspectRatio=True)
        
        # Captions
        c.setFont("Helvetica-Oblique", 10)
        c.setFillColor(colors.gray)
        c.drawString(50, img_y - 15, t["orig_img"])
        c.drawString(310, img_y - 15, t["heatmap_img"])

        # Disclaimer Footer
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(colors.red)
        c.drawString(50, 100, t["disclaimer_title"])
        
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.black)
        c.drawString(50, 85, t["disclaimer_text1"])
        c.drawString(50, 70, t["disclaimer_text2"])
        
        # Finalize
        c.showPage()
        c.save()
        
        buffer.seek(0)
        return StreamingResponse(
            buffer, 
            media_type="application/pdf", 
            headers={"Content-Disposition": "attachment; filename=MedOral_Report.pdf"}
        )
        
    except Exception as e:
        print(f"[ERROR] PDF generation failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

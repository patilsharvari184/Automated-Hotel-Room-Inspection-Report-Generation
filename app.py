from flask import Flask, render_template, request, redirect, url_for, send_file # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image    # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input    # type: ignore
import numpy as np  # type: ignore
import os
import datetime
import random
from reportlab.pdfgen import canvas # type: ignore
from reportlab.lib.pagesizes import letter  # type: ignore
from reportlab.lib.utils import ImageReader # type: ignore

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model1.h5'
model = load_model(MODEL_PATH)

# Helper functions
def generate_inspection_id():
    return f"A{random.randint(100, 999)}"

def predict_image(image_path):
    img = image.load_img(image_path, target_size=(500, 500))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    classes = model.predict(img_data)
    clean_prob, messy_prob = classes[0]
    prediction = "Clean" if clean_prob > messy_prob else "Messy"
    confidence = max(clean_prob, messy_prob) * 100
    return prediction, confidence

################# PDF Report Generation #################

def generate_pdf_report(data, image_path, output_pdf='static/reports/report.pdf'):
    """Creates a PDF inspection report with room image and classification results."""
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 50, "Cleanliness Inspection Report")

    # General Information
    c.setFont("Helvetica", 12)
    y_position = height - 80  # Initial Y position

    details = [
        f"Inspection ID: {data['inspection_id']}",
        f"Date of Inspection: {data['date']}",
        f"Property Name: {data['property_name']}",
        f"Property Region: {data['property_region']}",
        f"Property Type: {data['property_type']}",
        f"Service: {data['service']}",
        f"Room/Area Inspected: {data['room_area']}",
        f"Inspection Type: {data['inspection_type']}"
    ]

    for detail in details:
        c.drawString(100, y_position, detail)
        y_position -= 20

    # Cleanliness Classification Summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y_position - 20, "Cleanliness Classification Summary:")
    c.setFont("Helvetica", 12)
    y_position -= 40

    c.drawString(100, y_position, f"Overall Cleanliness Status: {data['cleanliness_status']}")
    c.drawString(100, y_position - 20, f"Confidence Score: {data['confidence_score']}")

    y_position -= 60  # Extra spacing before the image

    if os.path.exists(image_path):
        img = ImageReader(image_path)
        c.drawImage(img, 100, y_position - 200, width=300, height=200, preserveAspectRatio=True, mask='auto')

    c.save()
    return output_pdf

# Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filepath = os.path.join('static/uploads', file.filename)
        file.save(filepath)

        prediction, confidence = predict_image(filepath)
        report_data = {
            "inspection_id": generate_inspection_id(),
            "date": datetime.datetime.now().strftime("%d/%m/%Y"),
            "property_name": "Sample Hotel",
            "property_region": "Urban",
            "property_type": "Hotel",
            "service": "Room Cleaning",
            "room_area": "Room 302",
            "inspection_type": "Routine",
            "cleanliness_status": prediction,
            "confidence_score": f"{confidence:.2f}%"
        }

        report_path = generate_pdf_report(report_data, filepath)

        return render_template('result.html', prediction=prediction, confidence=confidence, report_path=report_path)

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/reports', exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)


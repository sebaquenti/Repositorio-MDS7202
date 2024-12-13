import gradio as gr
import requests

def predict_from_file(file):
    url = "http://backend:8000/predict/"
    files = {'file': file}
    response = requests.post(url, files=files)
    return response.json()

import gradio as gr
import requests
import pandas as pd

backend_url = "http://backend:8000"

def predict_from_csv(file):
    response = requests.post(f"{backend_url}/predict_csv/", files={"file": file})
    return response.json()["predictions"]

def predict_from_form(data):
    df = pd.DataFrame([data])
    response = requests.post(f"{backend_url}/predict/", json={"data": df.to_dict(orient="records")})
    return response.json()["predictions"]

with gr.Blocks() as demo:
    gr.Markdown("## Predicción de Modelo")

    with gr.Tab("Cargar CSV"):
        csv_input = gr.File(label="Sube un archivo CSV")
        csv_output = gr.Textbox(label="Predicciones")
        csv_button = gr.Button("Predecir")
        csv_button.click(predict_from_csv, inputs=csv_input, outputs=csv_output)

    with gr.Tab("Formulario Manual"):
        # Ajusta los campos según las características de tu modelo
        form_inputs = [
            gr.Textbox(label="Característica 1", name="feature1"),
            gr.Textbox(label="Característica 2", name="feature2"),
            # Añade más campos según sea necesario
        ]
        form_output = gr.Textbox(label="Predicciones")
        form_button = gr.Button("Predecir")
        form_button.click(predict_from_form, inputs=form_inputs, outputs=form_output)

demo.launch()

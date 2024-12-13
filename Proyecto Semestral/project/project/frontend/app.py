import gradio as gr
import pandas as pd
import requests

def predict_csv(file):
    response = requests.post("http://localhost:8000/predict_csv/", files={"file": file})
    return response.json()["predictions"]

def predict_manual(input_data):
    input_df = pd.DataFrame([input_data])
    response = requests.post("http://localhost:8000/predict/", json={"data": input_df.to_dict(orient="records")})
    return response.json()["predictions"]

with gr.Blocks() as demo:
    gr.Markdown("# Interfaz para Predicciones")
    
    with gr.Tab("Subir CSV"):
        file_input = gr.File(label="Sube un archivo CSV")
        csv_output = gr.Textbox(label="Predicciones")
        file_button = gr.Button("Predecir")
        file_button.click(predict_csv, inputs=file_input, outputs=csv_output)
    
    with gr.Tab("Ingresar Datos Manualmente"):
        feature1 = gr.Textbox(label="Característica 1")
        feature2 = gr.Textbox(label="Característica 2")
        manual_output = gr.Textbox(label="Predicciones")
        manual_button = gr.Button("Predecir")
        manual_button.click(
            predict_manual,
            inputs=[feature1, feature2],
            outputs=manual_output
        )

demo.launch()

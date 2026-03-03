# frontend/app.py
import gradio as gr
import requests
import os

API_URL = "http://localhost:9000"



def process_files(files):
    files_data = []

    for file in files:
        filename = os.path.basename(file.name)
        files_data.append(("files", (filename, open(file.name, "rb"))))

    try:
        # Subimos archivos
        response = requests.post(
            f"{API_URL}/upload",
            files=files_data
        )

        # Actualizamos la lista de archivos disponibles
        list_response = requests.get(f"{API_URL}/list_documents")
        list_data = list_response.json()

        return (
            gr.update(
                choices=list_data["documents"],
                value=list_data["documents"]
            ),
            "Files uploaded successfully"
        )

    finally:
        # cerrar todos los archivos abiertos
        for _, (_, f) in files_data:
            f.close()


def chat_response(message, history, selected_files):

    response = requests.post(
        f"{API_URL}/chat",
        json={
            "question": message,
            "selected_files": selected_files
        }
    )

    return response.json()["answer"]



def load_documents():
    try:
        response = requests.get(f"{API_URL}/list_documents")
        data = response.json()
        return gr.update(choices=data["documents"], value=data["documents"])
    except:
        return gr.update(choices=[], value=[])


with gr.Blocks() as demo:

    file_upload = gr.File(file_count="multiple")
    upload_btn = gr.Button("Upload")
    status = gr.Textbox()

    file_selector = gr.CheckboxGroup(label="Available files", choices=[])

    chatbot = gr.ChatInterface(
        fn=chat_response,
        additional_inputs=[file_selector]
    )

    upload_btn.click(
        fn=process_files,
        inputs=[file_upload],
        outputs=[file_selector, status]
    )
    
    demo.load(
    fn=load_documents,
    inputs=None,
    outputs=file_selector
    )


demo.launch()
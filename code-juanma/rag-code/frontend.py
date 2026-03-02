import gradio as gr
import requests
import os

API_URL = "http://127.0.0.1:8001"

# --- API communication functions ---

def load_existing_files_ui():
    """It runs when the page loads or when the refresh button is clicked.."""
    print("Requesting files from the backend...")
    try:
        response = requests.get(f"{API_URL}/files")
        response.raise_for_status()
        files = response.json().get("files", [])
        
        print(f"Files received from backend: {files}")
        
        return gr.update(choices=files, value=files)
    except Exception as e:
        print(f"Error loading files: {e}")
        return gr.update(choices=[], value=[])

def process_files_ui(files):
    if not files:
        return gr.update(), "No files were uploaded."

    print(f"Uploading {len(files)} files to the backend...")
    upload_data = [('files', (os.path.basename(f.name), open(f.name, 'rb'))) for f in files]
    
    try:
        response = requests.post(f"{API_URL}/upload", files=upload_data)
        response.raise_for_status()
        data = response.json()
        
        choices = data.get("processed_files", [])
        status = data.get("status_message", "")
        
        print(f"Upload completed. New list of files: {choices}")
        
        return gr.update(choices=choices, value=choices), status
    except Exception as e:
        print(f"Error uploading files: {e}")
        return gr.update(), f"Error connecting to backend: {e}"

def chat_response_ui(message, history, selected_files):
    if not message: return ""
    if not selected_files: return "Please select at least one file from the left panel."

    payload = {"message": message, "selected_files": selected_files}
    
    try:
        response = requests.post(f"{API_URL}/chat", json=payload)
        response.raise_for_status()
        return response.json().get("response", "Error reading response from backend.")
    except Exception as e:
        return f"Error connecting to backend: {e}"

def visualize_extended_context_ui():
    try:
        response = requests.get(f"{API_URL}/inspector")
        response.raise_for_status()
        return response.json().get("html", "")
    except Exception as e:
        return f"<p>Error connecting to backend: {e}</p>"

# --- Gradio interface ---

with gr.Blocks(title="RAG") as demo:
    gr.Markdown("# RAG DIA")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Documents")
            file_upload = gr.File(file_count="multiple", label="Upload new files")
            upload_btn = gr.Button("Process files", variant="primary")
            status_msg = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("### 2. Context active")
            btn_refresh_files = gr.Button("🔄 Refresh list of DB", variant="secondary", size="sm")
            file_selector = gr.CheckboxGroup(label="Files available in the database", choices=[])

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("Chatbot"):
                    chatbot = gr.ChatInterface(
                        fn=chat_response_ui,
                        additional_inputs=[file_selector],
                        description="Ask questions. The system will search in the selected documents."
                    )

                with gr.TabItem("Inspector"):
                    gr.Markdown("Review the exact fragments used for the last response.")
                    btn_refresh_context = gr.Button("Load context of the last response", variant="secondary")
                    html_viewer = gr.HTML(label="Chunks visualization")

                    btn_refresh_context.click(
                        fn=visualize_extended_context_ui,
                        inputs=[],
                        outputs=[html_viewer]
                    )

    # --- Events ---
    
    # Event 1: automatically upload files when opening the page
    demo.load(
        fn=load_existing_files_ui,
        inputs=[],
        outputs=[file_selector]
    )

    # Event 2: manual button to refresh the database list
    btn_refresh_files.click(
        fn=load_existing_files_ui,
        inputs=[],
        outputs=[file_selector]
    )

    # Event 3: upload new files
    upload_btn.click(
        fn=process_files_ui,
        inputs=[file_upload],
        outputs=[file_selector, status_msg]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), server_port=7860)
import requests
import gradio as gr

MODEL_API_URL = "http://localhost:8000/transcribe"


def transcribe_audio(audio_path):
    if audio_path is None:
        return "Please upload or record an audio file."

    try:
        with open(audio_path, "rb") as f:
            response = requests.post(
                MODEL_API_URL,
                files={"file": f},
                timeout=None  # allow long processing
            )

        if response.status_code != 200:
            return f"Error: {response.text}"

        return response.json().get("text", "No transcription returned.")

    except Exception as e:
        return f"Request failed: {str(e)}"


with gr.Blocks() as demo:
    gr.Markdown("# Voice to Text")

    audio_input = gr.Audio(type="filepath")
    output_text = gr.Textbox(label="Transcription", lines=15)
    transcribe_btn = gr.Button("Transcribe")

    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=audio_input,
        outputs=output_text
    )


def launch():
    demo.launch(server_name="0.0.0.0", server_port=7860)

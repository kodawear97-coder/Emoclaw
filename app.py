import gradio as gr
import os
from openai import OpenAI

# NVIDIA Client Setup
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

def chat_with_nemotron(message, history):
    if not NVIDIA_API_KEY:
        return "Error: NVIDIA_API_KEY not set!"
    try:
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            messages=[{"role": "user", "content": message}],
            temperature=0.5,
            max_tokens=1024,
            stream=False,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(title="Emoclaw AI Bot") as demo:
    gr.Markdown("# Emoclaw - AI-Powered Work Automation Bot")
    gr.ChatInterface(
        fn=chat_with_nemotron,
        examples=[
            "Kya hai Emoclaw?",
            "NVIDIA Nemotron kya hai?",
            "AI bot kaise banayein?"
        ],
        title="Chat with Emoclaw",
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

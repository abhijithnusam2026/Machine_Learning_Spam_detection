"""Hugging Face Spaces Gradio entrypoint.

The Docker deployment uses src.deployment.api_server, but non-Docker Gradio
Spaces expect a root-level app.py. Keep this thin wrapper so both deployment
modes use the same refactored implementation.
"""

from src.deployment.app import demo


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

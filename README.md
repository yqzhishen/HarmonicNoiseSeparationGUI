# HarmonicNoiseSeparationGUI

A simple WebUI for harmonic-noise separation of vocals, using ONNXRuntime for inference.

1. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Put the model files (*.onnx) in `models/` directory
3. Run the WebUI:
    ```bash
    python main.py [--port PORT] [--host HOST] [--work-dir WORK_DIR]
    ```
4. Open the WebUI in your browser (default: http://localhost:7861)

import pathlib
import time

import click
import gradio as gr
import onnxruntime as ort
from scipy.signal import resample

from inference import infer


class WebUI:
    def __init__(self, work_dir: pathlib.Path):
        self.work_dir = work_dir
        self.sessions = {}

    def get_session(self, model_rel_path):
        session = self.sessions.get(model_rel_path)
        if session is None:
            model_path = self.work_dir / model_rel_path
            session = ort.InferenceSession(model_path)
            self.sessions[model_rel_path] = session
        return session

    def infer_webui(
            self,
            model_rel_path,
            input_audio_tuple,
            chunk_length,
            batch_size
    ):
        if input_audio_tuple is None:
            return None, None, "Error: No input audio provided."
        sr, arr = input_audio_tuple
        if arr.ndim == 2:
            arr = arr[:, 0]
        waveform = (arr / 32768).astype('float32')
        original_samples = waveform.shape[0]
        if sr != 44100:
            waveform = resample(waveform, int(waveform.shape[0] * 44100 / sr))
        chunk_size = chunk_length * 44100
        overlap_size = max(chunk_size // 4, 16384)
        cut_size = max(chunk_size // 16, 4096)
        session = self.get_session(model_rel_path)
        if session is None:
            model_path = self.work_dir / model_rel_path
            session = ort.InferenceSession(model_path)
            self.sessions[model_rel_path] = session
        start = time.time()
        harmonic, noise = infer(
            session=session, waveform=waveform,
            chunk_size=chunk_size, overlap_size=overlap_size, cut_size=cut_size,
            batch_size=batch_size
        )
        end = time.time()
        if sr != 44100:
            harmonic = resample(harmonic, original_samples)
            noise = resample(noise, original_samples)
        harmonic = (harmonic * 32768).astype('int16')
        noise = (noise * 32768).astype('int16')
        rtf = (end - start) / (original_samples / 44100)
        return (sr, harmonic), (sr, noise), f"Cost: {end - start:.2f}s, RTF: {rtf:.2f}"


@click.command(help='Launch the WebUI for inference')
@click.option('--port', type=int, default=7861, help='Server port')
@click.option('--host', type=str, required=False, help='Server address')
@click.option(
    '--work-dir',
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=pathlib.Path
    ),
    default=pathlib.Path(__file__).parent / 'models',
    required=True, help='Working directory for the WebUI')
def webui(port, host, work_dir):
    choices = [
        p.relative_to(work_dir).as_posix()
        for p in work_dir.rglob('*.onnx')
    ]
    if len(choices) == 0:
        raise FileNotFoundError(f'No checkpoints found in {work_dir}.')
    inference = WebUI(work_dir)
    iface = gr.Interface(
        title="Harmonic-Noise Separation for Vocals",
        description="Submit a clean, non-instrumental vocal recording and separate the harmonic and noise parts.",
        theme="default",
        fn=inference.infer_webui,
        inputs=[
            gr.components.Dropdown(
                label="Model Checkpoint",
                choices=choices, value=choices[0],
                multiselect=False, allow_custom_value=False
            ),
            gr.components.Audio(label="Input Audio File", type="numpy"),
            gr.components.Slider(
                label='Chunk Length (seconds)', minimum=2, maximum=60, step=1, value=10
            ),
            gr.components.Slider(label='Batch Size', minimum=1, maximum=64, step=1, value=8),
        ],
        outputs=[
            gr.components.Audio(label="Output Harmonic", type="numpy"),
            gr.components.Audio(label="Output Noise", type="numpy"),
            gr.components.Label(label="Message"),
        ],
        concurrency_limit=10
    )
    iface.launch(server_port=port, server_name=host, share=False)


if __name__ == "__main__":
    webui()

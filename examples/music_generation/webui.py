import os
import sys
import torch
import gradio as gr
from inspiremusic.cli.inference import InspireMusicModel, env_variables
import torchaudio
import datetime
import hashlib
import tempfile
import shutil

# Get the project root directory (two levels up from current file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Available models
MODELS = [
    "InspireMusic-1.5B-Long",
    "InspireMusic-1.5B",
    "InspireMusic-Base",
    "InspireMusic-1.5B-24kHz",
    "InspireMusic-Base-24kHz"
]

# Default directories
AUDIO_PROMPT_DIR = os.path.join(PROJECT_ROOT, "demo/audio_prompts")
OUTPUT_AUDIO_DIR = os.path.join(PROJECT_ROOT, "demo/outputs")
PRETRAINED_MODELS_DIR = os.path.join(PROJECT_ROOT, "pretrained_models")

# Create necessary directories
os.makedirs(AUDIO_PROMPT_DIR, exist_ok=True)
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
os.makedirs(PRETRAINED_MODELS_DIR, exist_ok=True)

# Create a temporary directory for Gradio outputs
TEMP_DIR = tempfile.mkdtemp()

# Example prompts
DEMO_TEXT_PROMPTS = [
    "Jazz music with drum beats.",
    "A captivating classical piano performance, this piece exudes a dynamic and intense atmosphere, showcasing intricate and expressive instrumental artistry.",
    "A soothing instrumental piece blending elements of light music and pop, featuring a gentle guitar rendition.",
    "The instrumental rock piece features dynamic oscillations and wave-like progressions, creating an immersive and energetic atmosphere.",
    "The classical instrumental piece exudes a haunting and evocative atmosphere, characterized by its intricate guitar work and profound emotional depth.",
    "Experience a dynamic blend of instrumental electronic music with futuristic house vibes, featuring energetic beats and a captivating rhythm."
]

def check_model_files(model_dir):
    """Check if all required model files exist in the local directory."""
    required_files = ["llm.pt", "flow.pt", "music_tokenizer", "wavtokenizer", "inspiremusic.yaml"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing required model files in {model_dir}: {', '.join(missing_files)}\n"
            "Please make sure you have downloaded the model files to the correct directory.\n"
            f"Expected directory structure:\n"
            f"{PRETRAINED_MODELS_DIR}/\n"
            f"  ├── {os.path.basename(model_dir)}/\n"
            f"  │   ├── llm.pt\n"
            f"  │   ├── flow.pt\n"
            f"  │   ├── music_tokenizer\n"
            f"  │   ├── wavtokenizer\n"
            f"  │   └── inspiremusic.yaml"
        )
    return model_dir

def generate_filename():
    """Generate a unique filename using timestamp hash."""
    hash_object = hashlib.sha256(str(int(datetime.datetime.now().timestamp())).encode())
    return hash_object.hexdigest()

def trim_audio(audio_file, cut_seconds=5):
    """Trim audio file to specified length."""
    audio, sr = torchaudio.load(audio_file)
    num_samples = cut_seconds * sr
    cutted_audio = audio[:, :num_samples]
    output_path = os.path.join(TEMP_DIR, f"audio_prompt_{generate_filename()}.wav")
    torchaudio.save(output_path, cutted_audio, sr)
    return output_path

def get_model_args(
    task,
    text="",
    audio=None,
    model_name="InspireMusic-Base",
    chorus="intro",
    output_sample_rate=48000,
    max_generate_audio_seconds=30.0,
    time_start=0.0,
    time_end=30.0,
    trim=False,
    gpu=0
):
    """Prepare arguments for model inference."""
    if "24kHz" in model_name:
        output_sample_rate = 24000

    fast = output_sample_rate == 24000

    # Get model directory and check files
    model_dir = os.path.join(PRETRAINED_MODELS_DIR, model_name)
    model_dir = check_model_files(model_dir)

    args = {
        "task": task,
        "text": text,
        "audio_prompt": audio,
        "model_name": model_name,
        "chorus": chorus,
        "fast": fast,
        "fade_out": True,
        "trim": trim,
        "output_sample_rate": output_sample_rate,
        "min_generate_audio_seconds": 10.0,
        "max_generate_audio_seconds": max_generate_audio_seconds,
        "max_audio_prompt_length": 5.0,
        "model_dir": model_dir,
        "result_dir": TEMP_DIR,  # Use temporary directory for outputs
        "output_fn": generate_filename(),
        "format": "wav",
        "time_start": time_start,
        "time_end": time_end,
        "fade_out_duration": 1.0,
        "gpu": gpu
    }

    if args["time_start"] is None:
        args["time_start"] = 0.0
    args["time_end"] = args["time_start"] + args["max_generate_audio_seconds"]

    return args

def generate_music(args):
    """Generate music using InspireMusic model."""
    env_variables()
    model = InspireMusicModel(
        model_name=args["model_name"],
        model_dir=args["model_dir"],
        min_generate_audio_seconds=args["min_generate_audio_seconds"],
        max_generate_audio_seconds=args["max_generate_audio_seconds"],
        sample_rate=24000,
        output_sample_rate=args["output_sample_rate"],
        load_jit=True,
        load_onnx=False,
        fast=args["fast"],
        result_dir=args["result_dir"],
        gpu=args["gpu"]
    )

    output_path = model.inference(
        task=args["task"],
        text=args["text"],
        audio_prompt=args["audio_prompt"],
        chorus=args["chorus"],
        time_start=args["time_start"],
        time_end=args["time_end"],
        output_fn=args["output_fn"],
        max_audio_prompt_length=args["max_audio_prompt_length"],
        fade_out_duration=args["fade_out_duration"],
        output_format=args["format"],
        fade_out_mode=args["fade_out"],
        trim=args["trim"]
    )
    
    # Copy the output file to the permanent output directory
    if output_path and os.path.exists(output_path):
        filename = os.path.basename(output_path)
        permanent_path = os.path.join(OUTPUT_AUDIO_DIR, filename)
        shutil.copy2(output_path, permanent_path)
    
    return output_path

def cleanup_temp_files():
    """Clean up temporary files."""
    try:
        shutil.rmtree(TEMP_DIR)
    except Exception as e:
        print(f"Error cleaning up temporary files: {e}")

def text_to_music(text, model_name, chorus, output_sample_rate, max_generate_audio_seconds, gpu):
    """Generate music from text prompt."""
    args = get_model_args(
        task='text-to-music',
        text=text,
        model_name=model_name,
        chorus=chorus,
        output_sample_rate=output_sample_rate,
        max_generate_audio_seconds=max_generate_audio_seconds,
        gpu=gpu
    )
    return generate_music(args)

def music_continuation(text, audio, model_name, chorus, output_sample_rate, max_generate_audio_seconds, gpu):
    """Continue music from audio prompt."""
    args = get_model_args(
        task='continuation',
        text=text,
        audio=trim_audio(audio, cut_seconds=5),
        model_name=model_name,
        chorus=chorus,
        output_sample_rate=output_sample_rate,
        max_generate_audio_seconds=max_generate_audio_seconds,
        gpu=gpu
    )
    return generate_music(args)

def create_interface():
    """Create Gradio interface."""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # InspireMusic
        - Support music generation tasks with long-form and high audio quality, sampling rates up to 48kHz.
        - Github: https://github.com/FunAudioLLM/InspireMusic/
        - ModelScope Studio: https://modelscope.cn/studios/iic/InspireMusic
        - Available models: [InspireMusic-1.5B-Long](https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-Long), [InspireMusic-1.5B](https://huggingface.co/FunAudioLLM/InspireMusic-1.5B), [InspireMusic-Base](https://huggingface.co/FunAudioLLM/InspireMusic-Base), [InspireMusic-1.5B-24kHz](https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-24kHz), [InspireMusic-Base-24kHz](https://huggingface.co/FunAudioLLM/InspireMusic-Base-24kHz)
        - Currently only supports English text prompts.
        """)

        with gr.Row():
            with gr.Column():
                model_name = gr.Dropdown(
                    MODELS,
                    label="Model",
                    value="InspireMusic-1.5B-Long"
                )
                chorus = gr.Dropdown(
                    ["intro", "verse", "chorus", "outro"],
                    label="Chorus Mode",
                    value="intro"
                )
                output_sample_rate = gr.Dropdown(
                    [48000, 24000],
                    label="Output Sample Rate (Hz)",
                    value=48000
                )
                max_generate_audio_seconds = gr.Slider(
                    10, 300,
                    label="Audio Length (seconds)",
                    value=30
                )
                gpu = gr.Number(
                    label="GPU ID",
                    value=0,
                    precision=0
                )

        with gr.Tabs():
            with gr.TabItem("Text to Music"):
                with gr.Row():
                    text_input = gr.Textbox(
                        label="Text Prompt",
                        value="Experience soothing and sensual instrumental jazz with a touch of Bossa Nova, perfect for a relaxing restaurant or spa ambiance."
                    )
                with gr.Row():
                    t2m_button = gr.Button("Generate Music")
                with gr.Row():
                    t2m_output = gr.Audio(
                        label="Generated Music",
                        type="filepath",
                        autoplay=True
                    )
                t2m_examples = gr.Examples(
                    examples=DEMO_TEXT_PROMPTS,
                    inputs=[text_input]
                )

            with gr.TabItem("Music Continuation"):
                with gr.Row():
                    text_input_cont = gr.Textbox(
                        label="Text Prompt (Optional)",
                        value=""
                    )
                    audio_input = gr.Audio(
                        label="Audio Prompt",
                        type="filepath"
                    )
                with gr.Row():
                    cont_button = gr.Button("Continue Music")
                with gr.Row():
                    cont_output = gr.Audio(
                        label="Continued Music",
                        type="filepath",
                        autoplay=True
                    )

        t2m_button.click(
            text_to_music,
            inputs=[
                text_input,
                model_name,
                chorus,
                output_sample_rate,
                max_generate_audio_seconds,
                gpu
            ],
            outputs=t2m_output
        )

        cont_button.click(
            music_continuation,
            inputs=[
                text_input_cont,
                audio_input,
                model_name,
                chorus,
                output_sample_rate,
                max_generate_audio_seconds,
                gpu
            ],
            outputs=cont_output
        )

    return demo

def main():
    """Main function to run the Gradio interface."""
    try:
        demo = create_interface()
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860
        )
    finally:
        cleanup_temp_files()

if __name__ == "__main__":
    main() 
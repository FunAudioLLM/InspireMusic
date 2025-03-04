[//]: # (# InspireMusic)
<p align="center">
 <a href="https://github.com/FunAudioLLM/InspireMusic" target="_blank">
        <img alt="logo" src="./asset/logo.png" width="100%"></a>
</p>

[//]: # (<p align="center">)

[//]: # ( <a href="https://github.com/FunAudioLLM/InspireMusic" target="_blank">)

[//]: # (        <img alt="InspireMusic" src="https://svg-banners.vercel.app/api?type=origin&text1=Inspire%20Music🎶&text2=🤗%20A%20Fundamental%20Music%20Song%20Audio%20Generation%20Toolkit&width=800&height=210"></a>)

[//]: # (</p>)

<p align="center">
 <a href="https://funaudiollm.github.io/inspiremusic" target="_blank">
        <img alt="Demo" src="https://img.shields.io/badge/Demo-InspireMusic?labelColor=%20%23FDB062&label=InspireMusic&color=%20%23f79009"></a>
<a href="https://github.com/FunAudioLLM/InspireMusic" target="_blank">
        <img alt="Code" src="https://img.shields.io/badge/Code-InspireMusic?labelColor=%20%237372EB&label=InspireMusic&color=%20%235462eb"></a>

<a href="https://modelscope.cn/models/iic/InspireMusic-1.5B-Long" target="_blank">
        <img alt="Model" src="https://img.shields.io/badge/InspireMusic-Model-green"></a>

<a href="https://modelscope.cn/studios/iic/InspireMusic/summary" target="_blank">
        <img alt="Space" src="https://img.shields.io/badge/Spaces-ModelScope-pink?labelColor=%20%237b8afb&label=Spaces&color=%20%230a5af8"></a>

<a href="https://huggingface.co/spaces/FunAudioLLM/InspireMusic" target="_blank">
        <img alt="Space" src="https://img.shields.io/badge/HuggingFace-Spaces?labelColor=%20%239b8afb&label=Spaces&color=%20%237a5af8"></a>

<a href="http://arxiv.org/abs/2503.00084" target="_blank">
        <img alt="Paper" src="https://img.shields.io/badge/arXiv-Paper-green"></a>
<a href="https://github.com/FunAudioLLM/InspireMusic" target="_blank">

[//]: # (<a href="https://huggingface.co/FunAudioLLM/InspireMusic-Base" target="_blank">)

[//]: # (        <img alt="Model" src="https://img.shields.io/badge/Model-InspireMusic?labelColor=%20%23FDA199&label=InspireMusic&color=orange"></a>)

[//]: # (<a href="https://arxiv.org/abs/" target="_blank">)

[//]: # (        <img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv?labelColor=%20%23528bff&label=arXiv&color=%20%23155EEF"></a>)

[//]: # (<a href="https://github.com/FunAudioLLM/InspireMusic" target="_blank">)

[//]: # (        <img alt="Githube Star" src="https://img.shields.io/github/stars/FunAudioLLM/InspireMusic"></a>)

[//]: # (<a href="https://github.com/FunAudioLLM/InspireMusic/blob/main/asset/QR.jpg" target="_blank">)

[//]: # (        <img src="https://img.shields.io/badge/group%20chat-group?&labelColor=%20%235462eb&color=%20%235462eb" alt="chat on WeChat"></a>)
[//]: # (<a href="https://discord.gg/nSPpRU7fRr" target="_blank">)

[//]: # (        <img src="https://img.shields.io/badge/discord-chat?&labelColor=%20%235462eb&color=%20%235462eb" alt="chat on Discord"></a>)

[//]: # (    <a href="https://github.com/FunAudioLLM/InspireMusic" target="_blank">)

[//]: # (        <img alt="Static Badge" src="https://img.shields.io/badge/v0.1-version?logo=free&color=%20%23155EEF&label=version&labelColor=%20%23528bff"></a>)
[//]: # (<a href="https://github.com/FunAudioLLM/InspireMusic/graphs/commit-activity" target="_blank">)

[//]: # (<img alt="Commits last month" src="https://img.shields.io/github/commit-activity/m/FunAudioLLM/InspireMusic?labelColor=%20%2332b583&color=%20%2312b76a"></a>)

[//]: # (    <a href="https://github.com/FunAudioLLM/InspireMusic" target="_blank">)

[//]: # (        <img alt="Issues closed" src="https://img.shields.io/github/issues-search?query=repo%3AFunAudioLLM%2FInspireMusic%20is%3Aclosed&label=issues%20closed&labelColor=%20%237d89b0&color=%20%235d6b98"></a>)

[//]: # (    <a href="https://github.com/FunAudioLLM/InspireMusic/discussions/" target="_blank">)

[//]: # (        <img alt="Discussion posts" src="https://img.shields.io/github/discussions/FunAudioLLM/InspireMusic?labelColor=%20%239b8afb&color=%20%237a5af8"></a>)
</p>

InspireMusic is a fundamental AIGC toolkit and models designed for music, song, and audio generation using PyTorch. 

![GitHub Repo stars](https://img.shields.io/github/stars/FunAudioLLM/InspireMusic) Please support our community project 💖 by starring it on GitHub 加⭐支持 🙏 

[**Highlights**](#Highlights)
| [**News**](https://github.com/FunAudioLLM/InspireMusic#News) 
| [**Introduction**](#Introduction)
| [**Installation**](#Installation)
| [**Quick Start**](https://github.com/FunAudioLLM/InspireMusic#QuickStart)
| [**Tutorial**](https://github.com/FunAudioLLM/InspireMusic#Tutorial)
| [**Models**](https://github.com/FunAudioLLM/InspireMusic#Models)
| [**Contact**](https://github.com/FunAudioLLM/InspireMusic#Contact)


---
<a name="Highlights"></a>
## Highlights
**InspireMusic** focuses on music generation, song generation and audio generation.
- A unified framework for music/song/audio generation. Controllable with text prompts, music genres, music structures, etc.
- Support music generation tasks with high audio quality, with available sampling rates of 24kHz, 48kHz. 
- Support long-form audio generation.
- Convenient fine-tuning and inference. Support mixed precision training (BF16, FP16/FP32). Provide convenient fine-tuning and inference scripts and strategies, allowing users to easily fine-tune their music generation models.

<a name="News"></a>
## What's New 🔥
- 2025/03: InspireMusic [Technical Report](http://arxiv.org/abs/2503.00084) is available.
- 2025/02: Online demo is available on [ModelScope Space](https://modelscope.cn/studios/iic/InspireMusic/summary) and [HuggingFace Space](https://huggingface.co/spaces/FunAudioLLM/InspireMusic).
- 2025/01: Open-source [InspireMusic-Base](https://modelscope.cn/models/iic/InspireMusic/summary), [InspireMusic-Base-24kHz](https://modelscope.cn/models/iic/InspireMusic-Base-24kHz/summary), [InspireMusic-1.5B](https://modelscope.cn/models/iic/InspireMusic-1.5B/summary), [InspireMusic-1.5B-24kHz](https://modelscope.cn/models/iic/InspireMusic-1.5B-24kHz/summary), [InspireMusic-1.5B-Long](https://modelscope.cn/models/iic/InspireMusic-1.5B-Long/summary) models for music generation. Models are available on both ModelScope and HuggingFace. 
- 2024/12: Support to generate 48kHz audio with super resolution flow matching.
- 2024/11: Welcome to preview 👉🏻 [**InspireMusic Demos**](https://funaudiollm.github.io/inspiremusic) 👈🏻. We're excited to share this with you and are working hard to bring even more features and models soon. Your support and feedback mean a lot to us!
- 2024/11: We are thrilled to announce the open-sourcing of the **InspireMusic** [code repository](https://github.com/FunAudioLLM/InspireMusic) and [demos](https://funaudiollm.github.io/inspiremusic). **InspireMusic** is a unified framework for music, song, and audio generation, featuring capabilities such as text-to-music conversion, music structure, genre control, and timestamp management. InspireMusic stands out for its exceptional music generation and instruction-following abilities.

<a name="Introduction"></a>
## Introduction
> [!Note]
> This repo contains the algorithm infrastructure and some simple examples. Currently only support English text prompts.

> [!Tip]
> To explore the performance, please refer to [InspireMusic Demo Page](https://funaudiollm.github.io/inspiremusic). We will open-source better & larger models soon.

InspireMusic is a unified music, song and audio generation framework through the audio tokenization and detokenization process integrated with a large autoregressive transformer. The original motive of this toolkit is to empower the common users to innovate soundscapes and enhance euphony in research through music, song, and audio crafting. The toolkit provides both inference and training code for AI generative models that create high-quality music. Featuring a unified framework, InspireMusic incorporates autoregressive Transformer and conditional flow-matching modeling (CFM), allowing for the controllable generation of music, songs, and audio with both textual and structural music conditioning, as well as neural audio tokenizers. Currently, the toolkit supports text-to-music generation and plans to expand its capabilities to include text-to-song and text-to-audio generation in the future.

## InspireMusic
<p align="center">
  <table>
    <tr>
       <td style="text-align:center;">
      <img alt="Light" src="asset/InspireMusic.png" width="100%" />
    </tr>
    <tr>
      <td style="text-align:center;">
<b>Figure 1.</b> An overview of the InspireMusic framework. We introduce InspireMusic, a unified framework for music, song, and audio generation capable of producing high-quality 48kHz long-form audio. InspireMusic consists of three key components:

- **Dual Audio Tokenizers**:
The framework first converts raw audio waveforms into discrete tokens that are efficiently processed by the autoregressive model. We employ two tokenizers: WavTokenizer converts 24kHz audio into 75Hz discrete tokens, while Hifi-Codec transforms 48kHz audio into 150Hz latent features suited for our flow matching model.

- **Autoregressive Transformer**:
This component is trained using a next-token prediction approach on both text and audio tokens, enabling it to generate coherent and contextually relevant audio sequences.

- **Super-Resolution Flow Matching** Model:
An ODE-based diffusion model, specifically a super-resolution flow matching (SRFM) model, maps the lower-resolution audio tokens to latent features with a higher sampling rate. A vocoder then generates the final audio waveform from these enhanced latent features.

InspireMusic supports a range of tasks including text-to-music, music continuation, music reconstruction, and music super-resolution.
      </td>
    </tr>
  </table>
</p>

<a name="Installation"></a>
## Installation

### Clone

- Clone the repo
``` sh
git clone --recursive https://github.com/FunAudioLLM/InspireMusic.git
# If you failed to clone submodule due to network failures, please run the following command until success
cd InspireMusic
git submodule update --init --recursive
```

### Install from Source

InspireMusic requires Python 3.8, PyTorch 2.0.1, flash attention 2.6.2/2.6.3. For CUDA 11.x, you can install the dependencies with the following commands:

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:
``` sh
conda create -n inspiremusic python=3.8
conda activate inspiremusic
cd InspireMusic
# pynini is required by WeTextProcessing, use conda to install it as it can be executed on all platforms.
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
# install flash attention to speedup training, support version 2.6.2/2.6.3
pip install flash-attn --no-build-isolation
```

- Install within the package:
```sh
cd InspireMusic
# You can run to install the packages
python setup.py install
pip install flash-attn --no-build-isolation
```

We also recommend having `sox` or `ffmpeg` installed, either through your system or Anaconda:
```sh
# # Install sox
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel

# Install ffmpeg
# ubuntu
sudo apt-get install ffmpeg
# centos
sudo yum install ffmpeg
```

### Use Docker
Run the following command to build a docker image from Dockerfile provided.

```sh
docker build -t inspiremusic .
```

Run the following command to start the docker container in interactive mode.

```shell
docker run -ti --gpus all -v .:/workspace/InspireMusic inspiremusic
```

<a name="QuickStart"></a>
### Quick Start

Here is a quick example inference script for music generation. 
``` sh
cd InspireMusic
mkdir -p pretrained_models

# Download models
# ModelScope
git clone https://www.modelscope.cn/iic/InspireMusic-1.5B-Long.git pretrained_models/InspireMusic-1.5B-Long
# HuggingFace
git clone https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-Long.git pretrained_models/InspireMusic-1.5B-Long

cd examples/music_generation
# run a quick inference example
bash infer_1.5b_long.sh
```

Here is a quick start running script to run music generation task including data preparation pipeline, model training, inference. 
``` sh
cd InspireMusic/examples/music_generation/
bash run.sh
```

### One-line Inference
#### Text-to-music Task

One-line Shell script for text-to-music task.
``` sh
cd examples/music_generation
# with flow matching
# use one-line command to get a quick try
python -m inspiremusic.cli.inference

# custom the config like the following one-line command
python -m inspiremusic.cli.inference --task text-to-music -m "InspireMusic-1.5B-Long" -g 0 -t "Experience soothing and sensual instrumental jazz with a touch of Bossa Nova, perfect for a relaxing restaurant or spa ambiance." -c intro -s 0.0 -e 30.0 -r "exp/inspiremusic" -o output -f wav 

# without flow matching
python -m inspiremusic.cli.inference --task text-to-music -g 0 -t "Experience soothing and sensual instrumental jazz with a touch of Bossa Nova, perfect for a relaxing restaurant or spa ambiance." --fast True
```

Alternatively, you can run the inference with just a few lines of Python code.
```python
from inspiremusic.cli.inference import InspireMusicUnified
from inspiremusic.cli.inference import set_env_variables
if __name__ == "__main__":
  set_env_variables()
  model = InspireMusicUnified(model_name = "InspireMusic-1.5B-Long")
  model.inference("text-to-music", "Experience soothing and sensual instrumental jazz with a touch of Bossa Nova, perfect for a relaxing restaurant or spa ambiance.")
```

#### Music Continuation Task

One-line Shell script for music continuation task.
``` sh
cd examples/music_generation
# with flow matching
python -m inspiremusic.cli.inference --task continuation -g 0 -a audio_prompt.wav
# without flow matching
python -m inspiremusic.cli.inference --task continuation -g 0 -a audio_prompt.wav --fast True
```

Alternatively, you can run the inference with just a few lines of Python code.
```python
from inspiremusic.cli.inference import InspireMusicUnified
from inspiremusic.cli.inference import set_env_variables
if __name__ == "__main__":
  set_env_variables()
  model = InspireMusicUnified(model_name = "InspireMusic-1.5B-Long")
  # just use audio prompt
  model.inference("continuation", None, "audio_prompt.wav")
  # use both text prompt and audio prompt
  model.inference("continuation", "Continue to generate jazz music.", "audio_prompt.wav")
```
<a name="Models"></a>
## Models
### Download Model

We strongly recommend that you download our pretrained `InspireMusic model`.

If you are an expert in this field, and you are only interested in training your own InspireMusic model from scratch, you can skip this step.

``` sh
# git模型下载，请确保已安装git lfs
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/InspireMusic-1.5B-Long.git pretrained_models/InspireMusic
```

### Available Models
Currently, we open source the music generation models support 24KHz mono and 48KHz stereo audio. 
The table below presents the links to the ModelScope and Huggingface model hub. More models will be available soon.

| Model name                            | Model Links                                                                                                                                                                                                                                                                                                                                   | Remarks                                                                                            |
|---------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| InspireMusic-Base-24kHz               | [![model](https://img.shields.io/badge/ModelScope-Model-green.svg)](https://modelscope.cn/models/iic/InspireMusic-Base-24kHz/summary) [![model](https://img.shields.io/badge/HuggingFace-Model-green.svg)](https://huggingface.co/FunAudioLLM/InspireMusic-Base-24kHz)                                                                        | Pre-trained Music Generation Model, 24kHz mono, 30s                                                |
| InspireMusic-Base                     | [![model](https://img.shields.io/badge/ModelScope-Model-green.svg)](https://modelscope.cn/models/iic/InspireMusic/summary) [![model](https://img.shields.io/badge/HuggingFace-Model-green.svg)](https://huggingface.co/FunAudioLLM/InspireMusic-Base)                                                                                         | Pre-trained Music Generation Model, 48kHz, 30s                                                     |
| InspireMusic-1.5B-24kHz               | [![model](https://img.shields.io/badge/ModelScope-Model-green.svg)](https://modelscope.cn/models/iic/InspireMusic-1.5B-24kHz/summary) [![model](https://img.shields.io/badge/HuggingFace-Model-green.svg)](https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-24kHz)                                                                        | Pre-trained Music Generation 1.5B Model, 24kHz mono, 30s                                           |
| InspireMusic-1.5B                     | [![model](https://img.shields.io/badge/ModelScope-Model-green.svg)](https://modelscope.cn/models/iic/InspireMusic-1.5B/summary) [![model](https://img.shields.io/badge/HuggingFace-Model-green.svg)](https://huggingface.co/FunAudioLLM/InspireMusic-1.5B)                                                                                    | Pre-trained Music Generation 1.5B Model, 48kHz, 30s                                                |
| InspireMusic-1.5B-Long ⭐              | [![model](https://img.shields.io/badge/ModelScope-Model-green.svg)](https://modelscope.cn/models/iic/InspireMusic-1.5B-Long/summary) [![model](https://img.shields.io/badge/HuggingFace-Model-green.svg)](https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-Long)                                                                          | Pre-trained Music Generation 1.5B Model, 48kHz, support long-form music generation more than 5mins |
| InspireSong-1.5B                      | [![model](https://img.shields.io/badge/ModelScope-Model-lightgrey.svg)]() [![model](https://img.shields.io/badge/HuggingFace-Model-lightgrey.svg)]()                                                                                                                                                                                          | Pre-trained Song Generation 1.5B Model, 48kHz stereo                                               |
| InspireAudio-1.5B                     | [![model](https://img.shields.io/badge/ModelScope-Model-lightgrey.svg)]() [![model](https://img.shields.io/badge/HuggingFace-Model-lightgrey.svg)]()                                                                                                                                                                                          | Pre-trained Audio Generation 1.5B Model, 48kHz stereo                                              |
| Wavtokenizer[<sup>[1]</sup>](https://openreview.net/forum?id=yBlVlS2Fd9) (75Hz) | [![model](https://img.shields.io/badge/ModelScope-Model-green.svg)](https://modelscope.cn/models/iic/InspireMusic-1.5B-Long/file/view/master?fileName=wavtokenizer%252Fmodel.pt) [![model](https://img.shields.io/badge/HuggingFace-Model-green.svg)](https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-Long/tree/main/wavtokenizer)       | An extreme low bitrate audio tokenizer for music with one codebook at 24kHz audio.                 |
| Music_tokenizer (75Hz)                | [![model](https://img.shields.io/badge/ModelScope-Model-green.svg)](https://modelscope.cn/models/iic/InspireMusic-1.5B-24kHz/file/view/master?fileName=music_tokenizer%252Fmodel.pt) [![model](https://img.shields.io/badge/HuggingFace-Model-green.svg)](https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-24kHz/tree/main/music_tokenizer) | A music tokenizer based on HifiCodec<sup>[2]</sup> at 24kHz audio.                                 |
| Music_tokenizer (150Hz)               | [![model](https://img.shields.io/badge/ModelScope-Model-green.svg)](https://modelscope.cn/models/iic/InspireMusic-1.5B-Long/file/view/master?fileName=music_tokenizer%252Fmodel.pt) [![model](https://img.shields.io/badge/HuggingFace-Model-green.svg)](https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-Long/tree/main/music_tokenizer) | A music tokenizer based on HifiCodec at 48kHz audio.                                               |

Our models have been trained within a limited budget, and there is still significant potential for performance improvement. We are actively working on enhancing the model's performance.

<a name="Tutorial"></a>
## Basic Usage

At the moment, InspireMusic contains the training code and inference code for [music generation](https://github.com/FunAudioLLM/InspireMusic/tree/main/examples/music_generation). More tasks such as song generation and audio generation will be supported in future.

### Training

Here is an example to train LLM model, support FP16 training. 
```sh
torchrun --nnodes=1 --nproc_per_node=8 \
    --rdzv_id=1024 --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
    inspiremusic/bin/train.py \
    --train_engine "torch_ddp" \
    --config conf/inspiremusic.yaml \
    --train_data data/train.data.list \
    --cv_data data/dev.data.list \
    --model llm \
    --model_dir `pwd`/exp/music_generation/llm/ \
    --tensorboard_dir `pwd`/tensorboard/music_generation/llm/ \
    --ddp.dist_backend "nccl" \
    --num_workers 8 \
    --prefetch 100 \
    --pin_memory \
    --deepspeed_config ./conf/ds_stage2.json \
    --deepspeed.save_states model+optimizer \
    --fp16
```

Here is an example code to train flow matching model, does not support FP16 training.
```sh
torchrun --nnodes=1 --nproc_per_node=8 \
    --rdzv_id=1024 --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
    inspiremusic/bin/train.py \
    --train_engine "torch_ddp" \
    --config conf/inspiremusic.yaml \
    --train_data data/train.data.list \
    --cv_data data/dev.data.list \
    --model flow \
    --model_dir `pwd`/exp/music_generation/flow/ \
    --tensorboard_dir `pwd`/tensorboard/music_generation/flow/ \
    --ddp.dist_backend "nccl" \
    --num_workers 8 \
    --prefetch 100 \
    --pin_memory \
    --deepspeed_config ./conf/ds_stage2.json \
    --deepspeed.save_states model+optimizer
```

### Inference

Here is an example script to quickly do model inference.
``` sh
cd InspireMusic/examples/music_generation/
bash infer.sh
```

Here is an example code to run inference with normal mode, i.e., with flow matching model for text-to-music and music continuation tasks.
```sh
pretrained_model_dir = "pretrained_models/InspireMusic/"
for task in 'text-to-music' 'continuation'; do
  python inspiremusic/bin/inference.py --task $task \
      --gpu 0 \
      --config conf/inspiremusic.yaml \
      --prompt_data data/test/parquet/data.list \
      --flow_model $pretrained_model_dir/flow.pt \
      --llm_model $pretrained_model_dir/llm.pt \
      --music_tokenizer $pretrained_model_dir/music_tokenizer \
      --wavtokenizer $pretrained_model_dir/wavtokenizer \
      --result_dir `pwd`/exp/inspiremusic/${task}_test \
      --chorus verse \
      --min_generate_audio_seconds 8 \
      --max_generate_audio_seconds 30 
done
```

Here is an example code to run inference with fast mode, i.e., without flow matching model for text-to-music and music continuation tasks.
```sh
pretrained_model_dir = "pretrained_models/InspireMusic/"
for task in 'text-to-music' 'continuation'; do
  python inspiremusic/bin/inference.py --task $task \
      --gpu 0 \
      --config conf/inspiremusic.yaml \
      --prompt_data data/test/parquet/data.list \
      --flow_model $pretrained_model_dir/flow.pt \
      --llm_model $pretrained_model_dir/llm.pt \
      --music_tokenizer $pretrained_model_dir/music_tokenizer \
      --wavtokenizer $pretrained_model_dir/wavtokenizer \
      --result_dir `pwd`/exp/inspiremusic/${task}_test \
      --chorus verse \
      --fast \
      --min_generate_audio_seconds 8 \
      --max_generate_audio_seconds 30 
done
```
#### Hardware & Execution Time
On an H800 GPU, InspireMusic-1.5B-Long can generate 30 seconds audio in approximately 48 seconds, RTF around 1.6~1.8.

For normal mode, we recommend using hardware with at least 24GB of GPU memory for optimal inference. For fast mode, 16GB GPU memory is enough.

## Roadmap

- [x] 2024/12
  - [x] 75Hz InspireMusic-Base model for music generation
    
- [x] 2025/01
    - [x] Support to generate 48kHz
    - [x] 75Hz InspireMusic-1.5B model for music generation
    - [x] 75Hz InspireMusic-1.5B-Long model for long-form music generation

- [x] 2025/02
    - [x] Technical report v1
    - [x] Provide Dockerfile

- [ ] 2025/03
    - [ ] Support audio generation task
    - [ ] 75Hz InspireAudio model for audio generation

- [ ] 2025/05
    - [ ] Support song generation task
    - [ ] 75Hz InspireSong model for song generation

- [ ] TBD
    - [ ] 25Hz InspireMusic model
    - [ ] Runtime SDK 
    - [ ] Support 48kHz stereo audio
    - [ ] Streaming inference mode support
    - [ ] Support more diverse instruction mode, multi-lingual instructions
    - [ ] InspireSong trained with more multi-lingual data
    - [ ] More...

## Citation
```bibtex
@misc{zhang2025inspiremusic,
      title={InspireMusic: Integrating Super Resolution and Large Language Model for High-Fidelity Long-Form Music Generation}, 
      author={Chong Zhang and Yukun Ma and Qian Chen and Wen Wang and Shengkui Zhao and Zexu Pan and Hao Wang and Chongjia Ni and Trung Hieu Nguyen and Kun Zhou and Yidi Jiang and Chaohong Tan and Zhifu Gao and Zhihao Du and Bin Ma},
      year={2025},
      eprint={2503.00084},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2503.00084}, 
}
```

---
### Friend Links
Checkout some awesome Github repositories from Tongyi Lab, Alibaba Group.

<p align="center">
<a href="https://github.com/modelscope/ClearerVoice-Studio" target="_blank">
        <img alt="Demo" src="https://img.shields.io/badge/Repo | Space-ClearVoice?labelColor=&label=ClearVoice&color=green"></a>
<a href="https://github.com/FunAudioLLM/CosyVoice" target="_blank">
        <img alt="Demo" src="https://img.shields.io/badge/Repo | Space-CosyVoice?labelColor=&label=CosyVoice&color=green"></a>
<a href="https://github.com/FunAudioLLM/SenseVoice" target="_blank">
        <img alt="Demo" src="https://img.shields.io/badge/Repo | Space-SenseVoice?labelColor=&label=SenseVoice&color=green"></a>
</p>

<a name="Contact"></a>
## Community & Discussion
* Please support our community project 🌟 by starring it on GitHub 🙏
* Welcome to join our DingTalk and WeChat groups to share and discuss algorithms, technology, and user experience feedback. You may scan the following QR codes to join our official chat groups accordingly. 

<p align="center">
  <table>
    <tr>
      <td style="text-align:center;">
        <a href="./asset/QR.jpg"><img alt="FunAudioLLM in DingTalk" src="https://img.shields.io/badge/FunAudioLLM-DingTalk-d9d9d9"></a>
      </td>
      <td style="text-align:center;">
        <a href="./asset/QR.jpg"><img alt="InspireMusic in DingTalk" src="https://img.shields.io/badge/InspireMusic-DingTalk-d9d9d9"></a>
      </td>
      <td style="text-align:center;">
        <a href="./asset/QR.jpg"><img alt="InspireMusic in WeChat" src="https://img.shields.io/badge/InspireMusic-WeChat-d9d9d9"></a>
      </td>
    </tr>
    <tr>
       <td style="text-align:center;">
      <img alt="Light" src="./asset/dingding.png" width="33%" />
      <td style="text-align:center;">
      <img alt="Light" src="./asset/dingtalk.png" width="33%" />
      <td style="text-align:center;">
      <img alt="Light" src="./asset/QR.jpg" width="33%" />
      </td>
    </tr>
  </table>
</p>

* [Github Discussion](https://github.com/FunAudioLLM/InspireMusic/discussions). Best for sharing feedback and asking questions.
* [GitHub Issues](https://github.com/FunAudioLLM/InspireMusic/issues). Best for bugs you encounter using InspireMusic, and feature proposals.

## Acknowledge

1. We borrowed a lot of code from [CosyVoice<sup>[3]</sup>](https://github.com/FunAudioLLM/CosyVoice).
3. We borrowed a lot of code from [WavTokenizer](https://github.com/jishengpeng/WavTokenizer).
4. We borrowed a lot of code from [AcademiCodec](https://github.com/yangdongchao/AcademiCodec).
5. We borrowed a lot of code from [FunASR](https://github.com/modelscope/FunASR).
6. We borrowed a lot of code from [FunCodec](https://github.com/modelscope/FunCodec).
7. We borrowed a lot of code from [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS).
9. We borrowed a lot of code from [WeNet](https://github.com/wenet-e2e/wenet).

## References

[1] Shengpeng Ji, Ziyue Jiang, Wen Wang, Yifu Chen, Minghui Fang, Jialong Zuo, Qian Yang, Xize Cheng, Zehan Wang, Ruiqi Li, Ziang Zhang, Xiaoda Yang, Rongjie Huang, Yidi Jiang, Qian Chen, Siqi Zheng, Wen Wang, Zhou Zhao, WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling, The Thirteenth International Conference on Learning Representations, 2025.

[2] Yang, Dongchao, Songxiang Liu, Rongjie Huang, Jinchuan Tian, Chao Weng, and Yuexian Zou, Hifi-codec: Group-residual vector quantization for high fidelity audio codec, arXiv preprint arXiv:2305.02765, 2023.

[3] Du, Zhihao, Qian Chen, Shiliang Zhang, Kai Hu, Heng Lu, Yexin Yang, Hangrui Hu et al. Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens. arXiv preprint arXiv:2407.05407, 2024. 

## Disclaimer
The content provided above is for academic purposes only and is intended to demonstrate technical capabilities. Some examples are sourced from the internet. If any content infringes on your rights, please contact us to request its removal.

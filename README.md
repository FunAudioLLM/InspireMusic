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
 <a href="https://iris2c.github.io/InspireMusic" target="_blank">
        <img alt="Demo" src="https://img.shields.io/badge/Demo%20👈🏻-InspireMusic?labelColor=%20%23FDB062&label=InspireMusic&color=%20%23f79009"></a>
<a href="https://github.com/FunAudioLLM/InspireMusic" target="_blank">
        <img alt="Code" src="https://img.shields.io/badge/Code%20⭐-InspireMusic?labelColor=%20%237372EB&label=InspireMusic&color=%20%235462eb"></a>

<a href="https://huggingface.co/FunAudioLLM/InspireMusic-Base" target="_blank">
        <img alt="Model" src="https://img.shields.io/badge/InspireMusic-Model-lightgrey"></a>

<a href="https://arxiv.org/abs/" target="_blank">
        <img alt="Paper" src="https://img.shields.io/badge/arXiv-Paper-lightgrey"></a>
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

InspireMusic is a fundamental AIGC toolkit designed for music, song, and audio generation using the PyTorch library. 

![GitHub Repo stars](https://img.shields.io/github/stars/FunAudioLLM/InspireMusic) Please support our community project 💖 by starring it on GitHub 加⭐支持 🙏 

---
<a name="Highligts"></a>
## Highlights
**InspireMusic** focuses on music generation, song generation and audio generation.
- A unified framework for music/song/audio generation.
- Controllable with text prompts, music genres, music structures, etc.
- Convenient Fine-tuning and Inference: Provide convenient fine-tuning and inference scripts and strategies, allowing users to easily fine-tune their music generation models.
- Support text-to-music, music continuation tasks with high audio quality.

<a name="What's News"></a>
## What's New 🔥

[//]: # (- 2024/12: The [InspireMusic-Base]&#40;https://huggingface.co/FunAudioLLM/InspireMusic-Base&#41; voice understanding model is open-sourced, which offers high quality, diverse text style, music structure, music genre control capability.  )
- 2024/11: Welcome to preview 👉🏻 [**InspireMusic Demos**](https://iris2c.github.io/InspireMusic) 👈🏻. We're excited to share this with you and are working hard to bring even more features and models soon. Your support and feedback mean a lot to us!
- 2024/11: We are thrilled to announce the open-sourcing of the **InspireMusic** [code repository](https://github.com/FunAudioLLM/InspireMusic) and [demos](https://iris2c.github.io/InspireMusic). **InspireMusic** is a unified framework for music, song, and audio generation, featuring capabilities such as text-to-music conversion, music structure, genre control, and timestamp management. InspireMusic stands out for its exceptional music generation and instruction-following abilities.

## Introduction
> [!Note]
> This repo contains the algorithm infrastructure and some simple examples.

> [!Tip]
> To explore the performance, please refer to [InspireMusic Demo Page](https://iris2c.github.io/InspireMusic). We will open-source InspireMusic models and HuggingFace Space soon.

InspireMusic is a unified music, song and audio generation framework through the audio tokenization and detokenization process integrated with a large autoregressive transformer. The original motive of this toolkit is to empower the common users to innovate soundscapes and enhance euphony in research through music, song, and audio crafting. The toolkit provides both inference and training code for AI generative models that create high-quality music. Featuring a unified framework, InspireMusic incorporates autoregressive Transformer and conditional flow-matching modeling (CFM), allowing for the controllable generation of music, songs, and audio with both textual and structural music conditioning, as well as neural audio tokenizers. Currently, the toolkit supports text-to-music generation and plans to expand its capabilities to include text-to-song and text-to-audio generation in the future.

## Installation

### Clone

- Clone the repo
``` sh
git clone --recursive https://github.com/FunAudioLLM/InspireMusic.git
# If you failed to clone submodule due to network failures, please run the following command until success
cd InspireMusic
git submodule update --init --recursive
```

### Install
InspireMusic requires Python 3.8, PyTorch 2.1.0. To install InspireMusic, you can run one of the following:

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:
``` sh
conda create -n inspiremusic python=3.8
conda activate inspiremusic
cd InspireMusic
# pynini is required by WeTextProcessing, use conda to install it as it can be executed on all platforms.
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
# install flash attention to speedup training
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

## Models
### Download Model

We strongly recommend that you download our pretrained `InspireMusic model`.

If you are an expert in this field, and you are only interested in training your own InspireMusic model from scratch, you can skip this step.

``` sh
# git模型下载，请确保已安装git lfs
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/InspireMusic.git pretrained_models/InspireMusic-Base
```

### Available Models
Currently, we open source the music generation models only that supporting 24KHz mono channel audio. 
The table below presents the links to the ModelScope and Huggingface model hub. More models will be available soon.

| Model name            | Model Links                                                                                                                                                                               | Remarks                                        |
|------------------------------|-------------------------------------------------------|-------------|
| InspireMusic-Base     | [![model](https://img.shields.io/badge/ModelScope-Model-lightgrey.svg)](https://modelscope.cn/models/iic/InspireMusic/summary) [![model](https://img.shields.io/badge/HuggingFace-Model-lightgrey.svg)](https://huggingface.co/FunAudioLLM/InspireMusic-Base) | Pre-trained Music Generation Model, 24kHz mono |
| InspireMusic-1.5B     | [![model](https://img.shields.io/badge/ModelScope-Model-lightgrey.svg)]() [![model](https://img.shields.io/badge/HuggingFace-Model-lightgrey.svg)]()                                      | Pre-trained Music Generation 1.5B Model, 24kHz mono       |
| InspireSong-Base      | [![model](https://img.shields.io/badge/ModelScope-Model-lightgrey.svg)]() [![model](https://img.shields.io/badge/HuggingFace-Model-lightgrey.svg)]()                                      | Pre-trained Song Generation Base Model, 24kHz mono         |
| InspireSong-1.5B      | [![model](https://img.shields.io/badge/ModelScope-Model-lightgrey.svg)]() [![model](https://img.shields.io/badge/HuggingFace-Model-lightgrey.svg)]()                                      | Pre-trained Song Generation 1.5B Model, 24kHz mono         |
| InspireAudio-1.5B     | [![model](https://img.shields.io/badge/ModelScope-Model-lightgrey.svg)]() [![model](https://img.shields.io/badge/HuggingFace-Model-lightgrey.svg)]()                                      | Pre-trained Audio Generation 1.5B Model, 24kHz mono        |

## Basic Usage

At the moment, InspireMusic contains the training code and inference code for [music generation](https://github.com/FunAudioLLM/InspireMusic/tree/main/examples/music_generation). More tasks such as song generation and audio generation will be supported in future.

### Quick Start

Here is a quick start running script to do music generation task including data preparation pipeline, model training, inference. 
``` sh
cd InspireMusic/examples/music_generation/
bash run.sh
```

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
pretrained_model_dir = "./pretrained_models/InspireMusic/"
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
pretrained_model_dir = "./pretrained_models/InspireMusic/"
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

## Roadmap

- [ ] 2024/12
  - [ ] 75Hz InspireMusic base model for music generation
  - [ ] Support song generation task
    
- [ ] 2025/01
    - [ ] 75Hz InspireSong model for song generation

- [ ] 2025/02
    - [ ] Support audio generation task 
    - [ ] 75Hz InspireMusic-1.5B model for music generation

- [ ] 2025/03
    - [ ] 75Hz InspireAudio model for music and audio generation

- [ ] TBD

    - [ ] 25Hz InspireMusic model
    - [ ] Support 48kHz stereo audio
    - [ ] Streaming inference mode support
    - [ ] Support more instruction mode, multi-lingual instructions
    - [ ] InspireSong trained with more multi-lingual data
    - [ ] More...

---
### Friend Links
Checkout some awesome Github repositories from Speech Lab of Institute for Intelligent Computing, Alibaba Group.

<p align="center">
<a href="https://github.com/modelscope/ClearerVoice-Studio" target="_blank">
        <img alt="Demo" src="https://img.shields.io/badge/Repo | Space-ClearVoice?labelColor=&label=ClearVoice&color=green"></a>
<a href="https://github.com/FunAudioLLM/CosyVoice" target="_blank">
        <img alt="Demo" src="https://img.shields.io/badge/Repo | Space-CosyVoice?labelColor=&label=CosyVoice&color=green"></a>
<a href="https://github.com/FunAudioLLM/SenseVoice" target="_blank">
        <img alt="Demo" src="https://img.shields.io/badge/Repo | Space-SenseVoice?labelColor=&label=SenseVoice&color=green"></a>
</p>

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
        <a href="./asset/QR.jpg"><img alt="InspireMusic in WeChat" src="https://img.shields.io/badge/InspireMusic-WeChat-d9d9d9"></a>
      </td>
    </tr>
    <tr>
       <td style="text-align:center;">
      <img alt="Light" src="./asset/dingding.png" width="68%" />
      <td style="text-align:center;">
      <img alt="Light" src="./asset/QR.jpg" width="58%" />
      </td>
    </tr>
  </table>
</p>

* [Github Discussion](https://github.com/FunAudioLLM/InspireMusic/discussions). Best for sharing feedback and asking questions.
* [GitHub Issues](https://github.com/FunAudioLLM/InspireMusic/issues). Best for bugs you encounter using InspireMusic, and feature proposals.

## Acknowledge

1. We borrowed a lot of code from [CosyVoice](https://github.com/FunAudioLLM/CosyVoice).
3. We borrowed a lot of code from [WavTokenizer](https://github.com/jishengpeng/WavTokenizer).
4. We borrowed a lot of code from [AcademiCodec](https://github.com/yangdongchao/AcademiCodec).
5. We borrowed a lot of code from [FunASR](https://github.com/modelscope/FunASR).
6. We borrowed a lot of code from [FunCodec](https://github.com/modelscope/FunCodec).
7. We borrowed a lot of code from [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS).
9. We borrowed a lot of code from [WeNet](https://github.com/wenet-e2e/wenet).

## Disclaimer
The content provided above is for academic purposes only and is intended to demonstrate technical capabilities. Some examples are sourced from the internet. If any content infringes on your rights, please contact us to request its removal.

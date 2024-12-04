#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Chong Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""InspireMusic setup script."""

import os

from setuptools import find_packages
from setuptools import setup


requirements = {
    "install": [
		"setuptools",
		"conformer==0.3.2",
		"diffusers==0.27.2",
		"gdown==5.1.0",
		"gradio==4.32.2",
		"grpcio==1.57.0",
		"grpcio-tools==1.57.0",
		"hydra-core==1.3.2",
		"HyperPyYAML==1.2.2",
		"inflect==7.3.1",
		"librosa==0.10.2",
		"lightning==2.3.3",
		"matplotlib==3.7.5",
		"modelscope==1.15.0",
		"networkx==3.1",
		"omegaconf==2.3.0",
		"onnx==1.16.0",
		"protobuf==4.25",
		"pydantic==2.7.0",
		"rich==13.7.1",
		"soundfile==0.12.1",
		"tensorboard==2.14.0",
		"torch==2.0.1",
		"torchaudio==2.0.2",
		"uvicorn==0.30.0",
		"wget==3.2",
		"fastapi==0.111.0",
		"fastapi-cli==0.0.4",
		"WeTextProcessing==1.0.3",
		"accelerate",
		"huggingface-hub==0.25.2",
		"julius",
		"onnxruntime-gpu==1.16.0",
		"onnxruntime==1.16.0",
		"transformers",
    ],
    # train: The modules invoked when training only.
    "train": [
        "deepspeed==0.14.2",
    ],
    # all: The modules should be optionally installled due to some reason.
    #      Please consider moving them to "install" occasionally
    "all": [
        # NOTE(kamo): Append modules requiring specific pytorch version or torch>2.0
        "transformers",
        "openai-whisper==20231117",
    ],
    "setup": [
        "numpy",
    ],
    "test": [
        "pytest>=3.3.0",
    ],
}
requirements["all"].extend(requirements["train"])
requirements["test"].extend(requirements["train"])

install_requires = requirements["install"]
setup_requires = requirements["setup"]
tests_require = requirements["test"]
extras_require = {k: v for k, v in requirements.items() if k not in ["install", "setup"]}

dirname = os.path.dirname(__file__)
version_file = os.path.join(dirname, "inspiremusic", "version.txt")
with open(version_file, "r") as f:
    version = f.read().strip()
setup(
    name="inspiremusic",
    version=version,
    url="https://github.com/FunAudioLLM/InspireMusic.git",
    author="Speech Lab of Institute for Intelligent Computing, Alibaba Group",
    author_email="chong.zhang@alibaba-inc.com",
    description="InspireMusic: A Fundamental Music, Song and Audio Generation Toolkit",
    long_description=open(os.path.join(dirname, "README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="The MIT License",
    packages=find_packages(include=["inspiremusic*"]),
    package_data={"inspiremusic": ["version.txt"]},
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    python_requires=">=3.8.0",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "inspiremusic = inspiremusic.bin.inference:main",
            "inspiremusic-train = funasr.bin.train:main",
        ]
    },
)
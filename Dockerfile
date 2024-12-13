# Use Nvidia Cuda container base, sync the timezone to GMT, and install necessary package dependencies. Binaries are
# not available for some python packages, so pip must compile them locally. This is why g++, make, and python3.9-dev are
# included in the list below. Cuda 11.8 is used instead of 12 for backwards compatibility. Cuda 11.8 supports compute
# capability 3.5 through 9.0.
FROM nvidia/cuda:11.8.0-base-ubuntu20.04
ENV TZ=Etc/GMT
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt install -y --no-install-recommends \
    ffmpeg \
    g++ \
    make \
    git \
    libsndfile1 \
    python3.9-dev \
    python3.9-venv \
    wget \
    unzip

# Switch to a limited user
ARG LIMITED_USER=luna
RUN useradd --create-home --shell /bin/bash $LIMITED_USER
USER $LIMITED_USER

# Some Docker directives (such as COPY and WORKDIR) and linux command options (such as wget's directory-prefix option)
# do not expand the tilde (~) character to /home/<user>, so define a temporary variable to use instead.
ARG HOME_DIR=/home/$LIMITED_USER

# Download the pretrained GPT models.
RUN mkdir -p ~/hay_say/temp_downloads/GPT_SoVITS/pretrained_models/ && \
    wget https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/s1bert25hz-2kh-longer-epoch%3D68e-step%3D50232.ckpt --directory-prefix=$HOME_DIR/hay_say/temp_downloads/GPT_SoVITS/pretrained_models/ && \
    wget https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/s2D488k.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/GPT_SoVITS/pretrained_models/ && \
    wget https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/s2G488k.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/GPT_SoVITS/pretrained_models/ && \
    wget https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-hubert-base/config.json --directory-prefix=$HOME_DIR/hay_say/temp_downloads/GPT_SoVITS/pretrained_models/chinese-hubert-base/ && \
    wget https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-hubert-base/preprocessor_config.json --directory-prefix=$HOME_DIR/hay_say/temp_downloads/GPT_SoVITS/pretrained_models/chinese-hubert-base/ && \
    wget https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-hubert-base/pytorch_model.bin --directory-prefix=$HOME_DIR/hay_say/temp_downloads/GPT_SoVITS/pretrained_models/chinese-hubert-base/ && \
    wget https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-roberta-wwm-ext-large/config.json --directory-prefix=$HOME_DIR/hay_say/temp_downloads/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/ && \
    wget https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-roberta-wwm-ext-large/pytorch_model.bin --directory-prefix=$HOME_DIR/hay_say/temp_downloads/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/ && \
    wget https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-roberta-wwm-ext-large/tokenizer.json --directory-prefix=$HOME_DIR/hay_say/temp_downloads/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/

# Download the pretrained UVR5 models.
RUN mkdir -p ~/hay_say/temp_downloads/tools/uvr5/uvr5_weights/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2_all_vocals.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/tools/uvr5/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP3_all_vocals.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/tools/uvr5/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5_only_main_vocal.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/tools/uvr5/uvr5_weights/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoAggressive.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/tools/uvr5/uvr5_weights/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoDeReverb.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/tools/uvr5/uvr5_weights/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoNormal.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/tools/uvr5/uvr5_weights/ && \
    wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx --directory-prefix=$HOME_DIR/hay_say/temp_downloads/tools/uvr5/uvr5_weights/onnx_dereverb_By_FoxJoy/

# Download the pretrained G2PWModel (for Chinese generation)
RUN mkdir -p ~/hay_say/temp_downloads/G2PWModel/ && \
    wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip --directory-prefix=$HOME_DIR/hay_say/temp_downloads/G2PWModel/

# Create virtual environments for GPT-Sovits and Hay Say's gpt_so_vits_server.
RUN python3.9 -m venv ~/hay_say/.venvs/gpt_so_vits; \
    python3.9 -m venv ~/hay_say/.venvs/gpt_so_vits_server

# Install all python dependencies for GPT-Sovits that are needed for inference.
# Note: This is done *before* cloning the repository because the dependencies are likely to change less often than the
# GPT-Sovits code itself. Cloning the repo after installing the requirements helps the Docker cache optimize build time.
# See https://docs.docker.com/build/cache
# fastapi version <= 0.112.1 is required due to https://github.com/fastapi/fastapi/issues/12133
# numpy < 2 is necessary for the selected version of onnxruntime, which is required for generating output in non-english languages
RUN ~/hay_say/.venvs/gpt_so_vits/bin/pip install \
    --timeout=300 \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.3.0+cu118 \
    LangSegment==0.3.5 \
    gradio==4.24.0 \
    transformers==4.46.0 \
    librosa==0.9.2 \
    einops==0.8.0 \
    pytorch-lightning==2.4.0 \
    ffmpeg-python==0.2.0 \
    cn2an==0.5.22 \
    pypinyin==0.53.0 \
    jieba-fast==0.53 \
    fastapi==0.112.1 \
    wordsegment==1.3.1 \
    g2p-en==2.1.0 \
    pyopenjtalk==0.3.4 \
    onnxruntime==1.16.3 \
    numpy==1.23.4 \
    opencc==1.1.1 \
    jamo==0.4.1 \
    ko-pron==1.3 \
    g2pk2==0.0.3 \
    pyjyutping==1.0.0

## install the 'averaged_perceptron_tagger_eng' and 'cmudict' tokenizers
RUN ~/hay_say/.venvs/gpt_so_vits/bin/python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
RUN ~/hay_say/.venvs/gpt_so_vits/bin/python -c "import nltk; nltk.download('cmudict')"

# Install the dependencies for the GPT-Sovits interface code.
RUN ~/hay_say/.venvs/gpt_so_vits_server/bin/pip install --timeout=300 --no-cache-dir \
    hay_say_common==1.0.8 \
    jsonschema==4.19.1 \
    safetensors

# Clone a modified version of GPT-Sovits that works with the rest of Hay Say.
RUN git clone -b main --single-branch -q https://github.com/hydrusbeta/GPT-SoVITS ~/hay_say/gpt_so_vits

# Create links so that the inference_cli.py script can find the tools and GPT_SoVITS modules
RUN ln -s /home/luna/hay_say/gpt_so_vits/tools/ /home/luna/hay_say/gpt_so_vits/GPT_SoVITS/tools && \
    ln -s /home/luna/hay_say/gpt_so_vits/GPT_SoVITS/ /home/luna/hay_say/gpt_so_vits/GPT_SoVITS/GPT_SoVITS

## Clone the Hay Say interface code
RUN git clone -b main --single-branch -q https://github.com/hydrusbeta/gpt_so_vits_server ~/hay_say/gpt_so_vits_server

# Move the pretrained models to the expected directories.
RUN mv ~/hay_say/temp_downloads/GPT_SoVITS/pretrained_models/* ~/hay_say/gpt_so_vits/GPT_SoVITS/pretrained_models && \
    mkdir -p ~/hay_say/gpt_so_vits/tools/uvr5 && \
    mv ~/hay_say/temp_downloads/tools/uvr5/* ~/hay_say/gpt_so_vits/tools/uvr5 && \
    mkdir -p ~/hay_say/gpt_so_vits/GPT_SoVITS/text/G2PWModel && \
    unzip -j ~/hay_say/temp_downloads/G2PWModel/G2PWModel_1.1.zip -d ~/hay_say/gpt_so_vits/GPT_SoVITS/text/G2PWModel && \
    rm ~/hay_say/temp_downloads/G2PWModel/G2PWModel_1.1.zip

# Create directories that are used by the Hay Say interface code
RUN mkdir -p ~/hay_say/gpt_so_vits/output/ && \
    mkdir -p ~/hay_say/gpt_so_vits/input/

# Expose port 6581, the port that Hay Say uses for GPT-SoVits.
# Also expose port 9872, in case someone wants to use the original GPT-SoVits UI.
EXPOSE 6581
EXPOSE 9872

## Execute the Hay Say interface code
CMD ["/bin/sh", "-c", "~/hay_say/.venvs/gpt_so_vits_server/bin/python ~/hay_say/gpt_so_vits_server/main.py --cache_implementation file"]
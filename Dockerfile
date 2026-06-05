# ============================================================
# Imagem do projeto rank-fusion (XMTC)
# Base oficial do PyTorch com CUDA 11.8 — mesma versão de torch/CUDA
# que validamos funcionando na Brev (torch 2.1.0+cu118).
# ============================================================
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Evita perguntas interativas durante a instalação de pacotes do sistema
ENV DEBIAN_FRONTEND=noninteractive

# Dependências de sistema:
#   - git: necessário para instalar o xclib (pyxclib) direto do GitHub
#   - build-essential: o xclib compila extensões em C/Cython
RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Atualiza o pip e instala Cython + numpy ANTES (o xclib precisa deles para compilar)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir Cython numpy

# Instala as dependências do projeto (copiamos só o requirements.txt para o build)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Baixa os recursos do NLTK exigidos pelo retriv (tokenização / stopwords)
# em um caminho global, para já virem prontos dentro da imagem.
RUN python -m nltk.downloader -d /usr/share/nltk_data punkt punkt_tab stopwords

# Pasta de trabalho padrão (será espelhada pela montagem -v em tempo de execução)
WORKDIR /workspace

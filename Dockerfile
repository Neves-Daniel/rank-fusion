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

# Atualiza o pip e instala as ferramentas de build que o xclib e suas dependências
# precisam para compilar sem isolamento:
#   - Cython: exigido pelo próprio xclib (pyxclib)
#   - pybind11: exigido pelo fasttext (dependência do xclib)
#   - numpy: usado na compilação de extensões
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir Cython numpy pybind11

# Copia o requirements.txt para o build
COPY requirements.txt /tmp/requirements.txt

# O xclib (pyxclib) precisa do Cython para compilar, mas não o declara como
# dependência de build. Por isso instalamos ele SEPARADO e com --no-build-isolation,
# para que ele use o Cython/numpy já instalados acima.
RUN pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/kunaldahiya/pyxclib.git"

# Instala o restante das dependências (sem as linhas do xclib, já instalado acima)
RUN grep -v -i 'xclib' /tmp/requirements.txt > /tmp/req-rest.txt && \
    pip install --no-cache-dir -r /tmp/req-rest.txt

# Baixa os recursos do NLTK exigidos pelo retriv (tokenização / stopwords)
# em um caminho global, para já virem prontos dentro da imagem.
RUN python -m nltk.downloader -d /usr/share/nltk_data punkt punkt_tab stopwords

# Pasta de trabalho padrão (será espelhada pela montagem -v em tempo de execução)
WORKDIR /workspace

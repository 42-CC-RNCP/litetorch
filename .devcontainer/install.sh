#!/bin/bash
set -e

apt-get update && apt-get install -y \
  zsh git curl sudo build-essential libgl1-mesa-glx

sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" || true
chsh -s /bin/zsh

curl -sSL https://install.python-poetry.org | python3 -
export PATH="/root/.local/bin:$PATH"

cd /workspace
poetry config virtualenvs.in-project true
poetry install || true

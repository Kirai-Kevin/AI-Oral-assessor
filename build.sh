#!/usr/bin/env bash
# build.sh

# Install system dependencies
apt-get update
apt-get install -y python3-dev portaudio19-dev libasound2-dev libportaudio2 libportaudiocpp0

# Install Python dependencies
pip install --upgrade pip
pip install wheel setuptools
pip install --only-binary :all: pyaudio || pip install pyaudio

# Install the rest of your requirements
pip install -r requirements.txt
#!/bin/bash
apt-get update
apt-get install -y ffmpeg
streamlit run app.py --server.port 8000 --server.address 0.0.0.0

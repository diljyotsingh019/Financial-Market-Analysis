FROM python:3.10
WORKDIR /app 
COPY . /app

# Upgrade pip to the latest version
RUN pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install -r requirements.txt
CMD ["python", "src/pipeline/implementation_pipeline.py"]
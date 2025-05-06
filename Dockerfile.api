FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY api api
COPY src/model/utils.py src/model/utils.py
COPY src/inference.py src/inference.py
COPY saved_models saved_models
COPY data/preprocess.py data/preprocess.py

EXPOSE 5000
CMD ["python", "api/app.py"]

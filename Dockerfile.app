FROM python:3.10-slim

WORKDIR /app

COPY app app

RUN pip install streamlit requests

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
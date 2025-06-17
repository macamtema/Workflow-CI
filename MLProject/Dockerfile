# MLProject/Dockerfile

FROM python:3.11-slim

WORKDIR /app

COPY model/ /app/model/
COPY predict.py /app/predict.py

RUN pip install tensorflow

CMD ["python", "predict.py"]

FROM python:3.8
WORKDIR /app
COPY model.pkl /app/model.pkl
RUN pip install scikit-learn
COPY inference.py /app/inference.py
CMD ["python", "inference.py"]

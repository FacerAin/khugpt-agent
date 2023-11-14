FROM python:3.9

COPY ./app /app
COPY ./requirements.txt /requirements.txt

RUN pip install -r requirements.txt

EXPOSE 1009

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "1009"]

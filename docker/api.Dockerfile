FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY docker/requirements-api.txt /tmp/requirements.txt
RUN pip install --requirement /tmp/requirements.txt

RUN addgroup --system app && adduser --system --ingroup app app
COPY --chown=app:app api ./api

USER app
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

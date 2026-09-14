FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY docker/requirements-dashboard.txt /tmp/requirements.txt
RUN pip install --requirement /tmp/requirements.txt

RUN addgroup --system app && adduser --system --ingroup app app
COPY --chown=app:app dashboard ./dashboard

USER app
EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]

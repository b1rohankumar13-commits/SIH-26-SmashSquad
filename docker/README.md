# BustSentinel containers

This model-independent setup runs the FastAPI service and Streamlit dashboard.
Local dashboard outputs are mounted read-only from `outputs/`.

```powershell
docker compose up --build
```

- Dashboard: http://127.0.0.1:8501
- API documentation: http://127.0.0.1:8000/docs

Stop the services with `docker compose down`.

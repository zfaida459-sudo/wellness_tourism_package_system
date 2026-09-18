FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=appuser:appuser wellness_tourism/__init__.py wellness_tourism/schema.py wellness_tourism/predict.py ./wellness_tourism/
COPY --chown=appuser:appuser app.py ./
COPY --chown=appuser:appuser .streamlit/config.toml ./.streamlit/config.toml
RUN mkdir models .model-cache && chown appuser:appuser models .model-cache
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
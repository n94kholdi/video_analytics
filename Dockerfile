FROM python:3.11-slim

ARG VERSION=dev
LABEL org.opencontainers.image.title="video-analytics" \
      org.opencontainers.image.version="${VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIDEO_ANALYTICS_JOBS_DIR=/app/output/dashboard \
    VIDEO_ANALYTICS_DETECTOR_MODEL=/app/All_weights/Weights_final/HumanDetection_light_input_640.onnx \
    VIDEO_ANALYTICS_REID_MODEL=/app/All_weights/Weights_final/Tracking_osnet_x0_25_msmt17.onnx

WORKDIR /app

# ffmpeg converts OpenCV's output into browser-compatible H.264 video.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes ffmpeg libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Only files that declare dependencies go in before `pip install`, so
# editing app/ or configs/ below can't invalidate this layer and force
# every package to be re-downloaded on rebuild.
COPY pyproject.toml README.md ./
RUN python -m pip install --no-cache-dir ".[api]"

# Model weights change far less often than app code; copying them
# before the source keeps this layer cached across code-only rebuilds.
COPY All_weights/Weights_final/HumanDetection_light_input_640.onnx ./All_weights/Weights_final/HumanDetection_light_input_640.onnx
COPY All_weights/Weights_final/Tracking_osnet_x0_25_msmt17.onnx ./All_weights/Weights_final/Tracking_osnet_x0_25_msmt17.onnx

COPY app ./app
COPY configs ./configs

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/output/dashboard /app/output/outbox /app/outputs \
    && chown -R appuser:appuser /app/output /app/outputs

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

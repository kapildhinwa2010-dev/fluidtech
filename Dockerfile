# A reproducible environment for the whole project (CPU only).
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir "numpy>=1.23,<3" pytest \
 && python selftest.py \
 && python fluidlayer_single.py --selftest

CMD ["python", "examples/quickstart.py"]

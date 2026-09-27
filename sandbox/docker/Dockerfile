FROM python:3.12-slim
LABEL mycode.sandbox.protocol="v1"
COPY requirements.txt /opt/mycode-sandbox/requirements.txt
RUN pip install --no-cache-dir -r /opt/mycode-sandbox/requirements.txt \
    && useradd --uid 10001 --create-home sandbox
COPY bootstrap.py /opt/mycode-sandbox/bootstrap.py
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/src HOME=/tmp
USER 10001:10001
WORKDIR /workspace
ENTRYPOINT ["python", "/opt/mycode-sandbox/bootstrap.py"]

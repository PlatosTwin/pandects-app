# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      pkg-config \
      libmariadb-dev \
      nodejs \
      npm \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

LABEL fly_launch_runtime="flask"

WORKDIR /code

RUN mkdir -p /code/backend
COPY requirements.txt /code/backend/requirements.txt
RUN mkdir -p /code/backend/emails
COPY emails/package.json /code/backend/emails/package.json
COPY emails/package-lock.json /code/backend/emails/package-lock.json
RUN pip3 install -r /code/backend/requirements.txt
RUN npm ci --prefix /code/backend/emails

COPY . /code/backend

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "60", "backend.app:app"]

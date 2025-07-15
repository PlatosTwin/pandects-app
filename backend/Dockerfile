# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      pkg-config \
      libmariadb-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

LABEL fly_launch_runtime="flask"

WORKDIR /code

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8080

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8080"]

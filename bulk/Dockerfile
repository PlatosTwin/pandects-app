FROM python:3.12-slim-bookworm

# Install MariaDB client, mydumper, and system tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      gnupg \
      lsb-release \
    && curl -LsS https://downloads.mariadb.com/MariaDB/mariadb_repo_setup \
      | bash -s -- --mariadb-server-version=11.8 \
    && apt-get install -y --no-install-recommends \
      mariadb-client \
      mydumper \
      bash \
      gzip \
      openssl \
    && rm -rf /var/lib/apt/lists/*

# Install boto3 for S3/R2 interaction
RUN pip install --no-cache-dir boto3

# Copy the restore script
COPY restore_from_r2.py /app/restore_from_r2.py

# Default command (can be overridden, but useful for testing)
CMD ["python3", "/app/restore_from_r2.py"]

FROM python:3.12-alpine3.20

RUN apk add --no-cache \
      mariadb-client \
      curl \
      bash \
      gzip \
      openssl

RUN pip install --no-cache-dir boto3

COPY dump_and_upload.sh /app/dump_and_upload.sh
RUN chmod +x /app/dump_and_upload.sh

CMD ["/app/dump_and_upload.sh"]

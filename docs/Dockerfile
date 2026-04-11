# syntax = docker/dockerfile:1

# Keep in sync with package.json engines (Node 24.x)
ARG NODE_VERSION=24
FROM node:${NODE_VERSION}-slim AS build

WORKDIR /app/docs
ENV NODE_ENV="production"

COPY docs/package.json docs/yarn.lock ./
RUN corepack enable && yarn install --frozen-lockfile

COPY docs ./
COPY branding /app/branding
RUN mkdir -p /app/frontend/public
COPY frontend/public/openapi.yaml /app/frontend/public/openapi.yaml

RUN yarn build

FROM nginx:alpine

RUN rm -rf /usr/share/nginx/html/*
COPY docs/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/docs/build/. /usr/share/nginx/html/

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

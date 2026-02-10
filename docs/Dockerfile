# syntax = docker/dockerfile:1

ARG NODE_VERSION=20.5.0
FROM node:${NODE_VERSION}-slim AS build

WORKDIR /app/docs
ENV NODE_ENV="production"

COPY docs/package.json docs/yarn.lock ./
RUN corepack enable && yarn install --frozen-lockfile

COPY docs ./
COPY branding ../../branding
COPY frontend/public/openapi.yaml ../../frontend/public/openapi.yaml

RUN yarn build

FROM nginx:alpine

RUN rm -rf /usr/share/nginx/html/*
COPY docs/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/docs/build/. /usr/share/nginx/html/

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

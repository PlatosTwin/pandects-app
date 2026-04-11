# syntax = docker/dockerfile:1

# Keep in sync with package.json engines (Node 24.x)
ARG NODE_VERSION=24
FROM node:${NODE_VERSION}-slim AS base

LABEL fly_launch_runtime="Vite"

# Vite app lives here
WORKDIR /app/frontend

# Set production environment
ENV NODE_ENV="production"


# Throw-away build stage to reduce size of final image
FROM base AS build

# Vite env vars are baked in at build time; pass as Docker build args (configured in fly.toml).
ARG VITE_API_BASE_URL
ENV VITE_API_BASE_URL="${VITE_API_BASE_URL}"

# Install packages needed to build node modules
RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y build-essential node-gyp pkg-config python-is-python3

# Install node modules
COPY frontend/package-lock.json frontend/package.json ./
RUN npm ci --include=dev

# Copy application code and the repo-root license consumed by the frontend build
COPY frontend ./
COPY LICENSE.md /app/LICENSE.md
COPY branding /app/branding

# Build application
RUN npm run build

# Remove development dependencies
RUN npm prune --omit=dev


# Final stage for app image
FROM nginx

# clear out nginx’s html folder
RUN rm -rf /usr/share/nginx/html/*

# copy spa build into nginx root
COPY --from=build /app/frontend/dist/spa/. /usr/share/nginx/html/

# overwrite default config with an SPA-friendly one
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD [ "nginx", "-g", "daemon off;" ]

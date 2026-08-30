FROM node:20.19-bookworm-slim AS frontend-build

WORKDIR /build/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


FROM python:3.11-slim

# The production image serves the prebuilt SPA from FastAPI. Node.js and npm
# remain in the build stage and are not part of the runtime attack surface.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential pkg-config gosu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 10001 scai \
    && mkdir -p /app \
    && chown scai:scai /app

WORKDIR /app

COPY --chown=scai:scai . .
COPY --from=frontend-build --chown=scai:scai \
    /build/frontend/dist /app/frontend/dist

ENV SCAI_FRONTEND_DIST=/app/frontend/dist

RUN chmod +x docker-entrypoint.sh

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD curl -f http://127.0.0.1:3000/api/health || exit 1

EXPOSE 3000

ENTRYPOINT ["./docker-entrypoint.sh"]

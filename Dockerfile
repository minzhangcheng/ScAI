FROM python:3.11-slim

# Install Node.js, npm, and curl (used for backend health check)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl nodejs npm build-essential pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN chmod +x docker-entrypoint.sh

EXPOSE 3000

ENTRYPOINT ["./docker-entrypoint.sh"]

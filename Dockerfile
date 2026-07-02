FROM python:3.11-slim

# Install Node.js, npm, curl, and gosu (privilege-dropping helper)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl nodejs npm build-essential pkg-config gosu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build-time pip layer: install dependencies first so PyPI issues at boot
# don't break startup (the /dep first-boot install remains as overlay).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build frontend for production
RUN cd frontend && npm install && npm run build && cd ..

# Create non-root user; chown everything so the entrypoint (running as root)
# can chown /dep and /data, then drop privileges via gosu.
RUN useradd -m -u 10001 scai \
    && chown -R scai:scai /app \
    && chmod +x docker-entrypoint.sh

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD curl -f http://127.0.0.1:8000/api/config/status || exit 1

EXPOSE 3000

ENTRYPOINT ["./docker-entrypoint.sh"]

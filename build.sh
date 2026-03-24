#!/bin/bash
set -e

IMAGE="ghcr.io/minzhangcheng/scai"
VERSION=""
PUSH=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./build.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --version TAG   Image version tag (default: git tag or 'latest')"
            echo "  --push          Push image to ghcr.io after build"
            echo "  -h, --help      Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine version from git tag if not specified
if [ -z "$VERSION" ]; then
    VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "latest")
fi

echo "Building $IMAGE:$VERSION ..."

docker build -t "$IMAGE:$VERSION" -t "$IMAGE:latest" .

echo ""
echo "Built: $IMAGE:$VERSION"
echo "Built: $IMAGE:latest"

if [ "$PUSH" = true ]; then
    echo ""
    echo "Pushing to ghcr.io..."
    docker push "$IMAGE:$VERSION"
    docker push "$IMAGE:latest"
    echo "Pushed successfully."
fi

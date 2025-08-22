#!/bin/bash

echo "Building and running whisper.cpp proxy service..."

# Build the docker image
docker-compose build

# Run the containers
docker-compose up -d

echo "Proxy service is running on port 8080"
echo "Usage example:"
echo "curl -X POST http://localhost:8080/inference \\"
echo "  -F file=@/path/to/audio.wav \\"
echo "  -F response_format=json"

# Show container logs
docker-compose logs -f

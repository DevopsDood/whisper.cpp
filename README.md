# Whisper.cpp Proxy Service

A minimal microservice that runs two instances of whisper.cpp with a proxy that alternates between them and restarts one every 5th transcription to avoid the bug that causes a replay of prior transcription failure after ~10-20 transcriptions.

## Architecture

The solution consists of:
1. Two whisper.cpp servers running on ports 8081 and 8082
2. A Python proxy server that routes requests between them
3. Load balancing with restart logic every 5th request to the same server

## Features

- **Alternating Load Balancing**: Requests are routed alternately between two whisper.cpp servers
- **Automatic Restart Logic**: Each server is automatically restarted after 5 transcription requests to prevent bugs
- **Health Checks**: Health endpoints to monitor server status
- **Dockerized**: Easy deployment with Docker Compose

## Quick Start

1. Build and run the service:
```bash
./run.sh
```

2. Make a transcription request:
```bash
curl -X POST http://localhost:8080/inference \
  -F file=@/path/to/audio.wav \
  -F response_format=json
```

## Configuration

The service can be configured by modifying:
- `Dockerfile` - for build settings and dependencies
- `proxy.py` - for server ports, model paths, and restart logic  
- `docker-compose.yml` - for container configuration

## How It Works

1. The proxy starts two whisper.cpp server instances
2. Each incoming request is routed to the next available server (alternating)
3. Each server tracks its own request count
4. When a server reaches 5 requests, it's automatically restarted
5. The proxy maintains state and handles request forwarding properly

## Requirements

- Docker and Docker Compose
- At least 8GB of RAM (for whisper.cpp models)
- Linux/macOS/Windows with Docker support

## Model Download

The service automatically downloads the `ggml-large-v3-turbo.bin` model from Hugging Face on first run.

## NVIDIA Support

For NVIDIA systems, the proxy automatically detects the presence of an NVIDIA GPU and ensures that 
the correct ggml model (non-CoreML) is used for optimal performance. This branch contains specific 
configurations to ensure compatibility with NVIDIA hardware.

## Troubleshooting

If you encounter issues:

1. Ensure Docker and Docker Compose are installed
2. Check that your system has sufficient memory (16GB+ recommended)
3. Verify the audio files are in a supported format (WAV, MP3, etc.)

## License

MIT

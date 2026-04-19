# Docker Build & Deployment Guide

## Quick Start with Docker Compose

### Build and start all services:
```bash
docker-compose up --build
```

### Start services in background:
```bash
docker-compose up -d
```

### Stop services:
```bash
docker-compose down
```

### View logs:
```bash
docker-compose logs -f app
```

## Building Docker Image Manually

### Build the image:
```bash
docker build -t resume-screening:latest .
```

### Build with a specific Python version tag:
```bash
docker build -t resume-screening:1.0 .
```

## Running Docker Container Manually

### Run with default settings:
```bash
docker run -p 5000:5000 resume-screening:latest
```

### Run with environment variables:
```bash
docker run -p 5000:5000 \
  -e MONGODB_URI=mongodb://host.docker.internal:27017 \
  -e OPENAI_API_KEY=your_key \
  resume-screening:latest
```

### Run with .env file:
```bash
docker run -p 5000:5000 --env-file .env resume-screening:latest
```

### Run interactively:
```bash
docker run -it -p 5000:5000 resume-screening:latest /bin/bash
```

## Container Registry Deployment

### Tag for registry:
```bash
docker tag resume-screening:latest myregistry.azurecr.io/resume-screening:latest
```

### Push to registry:
```bash
docker push myregistry.azurecr.io/resume-screening:latest
```

## Useful Docker Commands

### List images:
```bash
docker images
```

### List running containers:
```bash
docker ps
```

### View container logs:
```bash
docker logs <container_id>
```

### Check health status:
```bash
docker ps --no-trunc
```

### Remove image:
```bash
docker rmi resume-screening:latest
```

### Remove all stopped containers:
```bash
docker container prune
```

## Environment Variables

The application reads configuration from environment variables. Key variables:

- `FLASK_HOST` - Flask server host (default: 0.0.0.0)
- `FLASK_PORT` - Flask server port (default: 5000)
- `FLASK_DEBUG` - Debug mode (default: false)
- `MONGODB_URI` - MongoDB connection string
- `MONGODB_DATABASE` - MongoDB database name
- `OPENAI_API_KEY` - OpenAI API key (if using GPT models)
- `TALENTMATCH_API_URL` - TalentMatch API endpoint
- `TALENTMATCH_API_KEY` - TalentMatch API key

## Performance Notes

- The multi-stage build keeps the final image size minimal by excluding build dependencies
- GPU support: To use CUDA-enabled PyTorch, use a different base image:
  ```dockerfile
  FROM pytorch/pytorch:2.0-cuda11.8-cudnn8-runtime
  ```
- Memory: The ML models require significant RAM. Ensure your Docker daemon has adequate resources allocated.

## Troubleshooting

### Permission denied error:
Use `docker ps` to check if the Docker daemon is running.

### Port already in use:
```bash
docker run -p 5001:5000 resume-screening:latest
```

### Container exits immediately:
Check logs: `docker logs <container_id>`

### Health check failing:
Ensure the `/health` endpoint is properly implemented and the app is fully started.

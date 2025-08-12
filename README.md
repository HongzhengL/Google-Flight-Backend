# GPT-OSS-20B API Service

A high-performance, OpenAI-compatible API service for the GPT-OSS-20B model, deployed on Google Cloud Run with GPU support.

## Overview

This service provides a production-ready API for GPT-OSS-20B (20.9 billion parameter MoE model) with streaming support and OpenAI-compatible endpoints. The model uses MXFP4 quantization to fit within Cloud Run's NVIDIA L4 GPU constraints while maintaining high performance.

### Key Features

- **OpenAI-Compatible API**: Full compatibility with OpenAI's chat completions and completions endpoints
- **Streaming Support**: Real-time token-by-token streaming via Server-Sent Events
- **GPU Acceleration**: Optimized for NVIDIA L4 GPUs on Google Cloud Run
- **Auto-scaling**: Scale-to-zero for cost efficiency, auto-scale up to 5 instances
- **128k Context**: Full 128,000 token context length support
- **Production Ready**: Health checks, monitoring, structured logging, and error handling

## API Endpoints

### Core Endpoints

- `POST /v1/chat/completions` - OpenAI-compatible chat completions with streaming
- `POST /v1/completions` - Text completions with streaming support
- `GET /v1/models` - List available models
- `GET /health` - Basic health check
- `GET /ready` - Readiness probe with system status
- `GET /livez` - Liveness probe for Kubernetes/Cloud Run
- `GET /docs` - Interactive API documentation (Swagger UI)

### Example Usage

#### Chat Completion (Streaming)
```bash
curl -X POST "https://your-service-url/v1/chat/completions" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [
      {"role": "user", "content": "Explain quantum computing briefly"}
    ],
    "stream": true,
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

#### Non-Streaming Chat
```bash
curl -X POST "https://your-service-url/v1/chat/completions" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is machine learning?"}
    ],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

## Quick Start

### Prerequisites

- Google Cloud Project with billing enabled
- Docker installed locally
- gcloud CLI configured and authenticated
- NVIDIA Docker runtime (for local GPU development)

### Local Development

1. **Clone and Setup**
   ```bash
   git clone <your-repo-url>
   cd gpt-oss-20b-api
   cp .env.example .env
   ```

2. **Configure Environment**
   Edit `.env` file:
   ```bash
   HUGGINGFACE_TOKEN=your_hf_token_here  # Optional but recommended
   MAX_NUM_SEQS=4                        # Concurrent requests
   GPU_MEMORY_UTILIZATION=0.90           # GPU memory usage
   LOG_LEVEL=INFO
   ```

3. **Run with Docker Compose** (requires NVIDIA Docker)
   ```bash
   docker-compose up --build
   ```

4. **Test the API**
   ```bash
   curl http://localhost:8080/health
   curl http://localhost:8080/ready
   ```

### Cloud Deployment

#### Option 1: Quick Deploy Script
```bash
export PROJECT_ID=your-gcp-project-id
export REGION=us-central1
./deploy.sh
```

#### Option 2: Manual Deployment
```bash
# Set your project
gcloud config set project your-gcp-project-id

# Enable required services
gcloud services enable cloudbuild.googleapis.com run.googleapis.com artifactregistry.googleapis.com

# Create Artifact Registry repository
gcloud artifacts repositories create cloud-run-source-deploy \
    --repository-format=docker --location=us-central1

# Build and deploy
gcloud builds submit --config cloudbuild.yaml \
    --substitutions _SERVICE_NAME=gpt-oss-20b-api,_REGION=us-central1
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGINGFACE_TOKEN` | None | Optional HF token for private models |
| `MODEL_NAME` | `gpt-oss-20b` | Model repository name |
| `MAX_MODEL_LEN` | `128000` | Maximum context length |
| `MAX_NUM_SEQS` | `4` | Concurrent request limit |
| `GPU_MEMORY_UTILIZATION` | `0.90` | GPU memory usage fraction |
| `QUANTIZATION` | `fp8` | Model quantization (fp8, None) |
| `DTYPE` | `auto` | Model data type |
| `LOG_LEVEL` | `INFO` | Logging level |

### Model Configuration

The service automatically downloads GPT-OSS-20B from Hugging Face on first startup. The model is cached in `/tmp/models` during container lifetime.

**Resource Requirements:**
- GPU: NVIDIA L4 (24GB VRAM) - required
- Memory: 32GB system RAM - recommended
- CPU: 8 vCPUs - recommended
- Storage: ~15GB for model weights

## Performance Tuning

### Concurrency Settings
- `MAX_NUM_SEQS=4`: Handle up to 4 concurrent requests (adjust based on model size and GPU memory)
- Cloud Run `--concurrency=4`: Matches vLLM concurrency for optimal performance

### Memory Optimization
- `GPU_MEMORY_UTILIZATION=0.90`: Use 90% of GPU memory for model and KV cache
- Reduce for lower memory models or to handle longer contexts

### Context Length vs. Throughput
- Shorter contexts enable higher concurrency
- 128k context requires significant memory allocation
- Monitor GPU memory usage and adjust accordingly

## Monitoring and Observability

### Health Checks
- `/health`: Basic service health (always returns 200 when running)
- `/ready`: Detailed system status (GPU, model loading, memory usage)
- `/livez`: Liveness probe for container orchestrators

### Logging
- Structured JSON logging to stdout
- Request/response logging for debugging
- GPU utilization and performance metrics

### Metrics
Access metrics at Cloud Run console or via:
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=gpt-oss-20b-api"
```

## Cost Optimization

### Scaling Strategy
- **Scale-to-zero**: No charges when idle
- **Min instances**: 0 (serverless)
- **Max instances**: 5 (adjustable)
- **Concurrency**: 4 concurrent requests per instance

### Usage Patterns
- GPU costs only during active inference
- Model loading adds ~30-60 seconds to cold starts
- Keep-warm instances for consistent latency (set min-instances > 0)

### Cost Estimates (US Central1)
- NVIDIA L4 GPU: ~$0.70/hour when running
- 32GB Memory: ~$0.06/hour when running
- 8 vCPU: ~$0.24/hour when running
- **Total**: ~$1.00/hour during active inference

## Troubleshooting

### Common Issues

1. **Model Download Fails**
   - Check Hugging Face token if using private models
   - Verify internet connectivity and Hugging Face Hub access
   - Check disk space in `/tmp/models`

2. **GPU Not Available**
   - Ensure Cloud Run service has GPU enabled
   - Verify NVIDIA Docker runtime for local development
   - Check CUDA availability in logs

3. **Out of Memory Errors**
   - Reduce `GPU_MEMORY_UTILIZATION` to 0.80 or lower
   - Decrease `MAX_NUM_SEQS` for lower memory usage
   - Monitor GPU memory via `/ready` endpoint

4. **Slow Responses**
   - Cold starts require model loading (30-60 seconds)
   - Consider min-instances > 0 for consistent latency
   - Check concurrent request limits

5. **Authentication Errors**
   - Use `gcloud auth print-access-token` for Bearer token
   - Ensure proper IAM permissions for Cloud Run
   - Verify service is deployed with `--no-allow-unauthenticated`

### Debug Commands

```bash
# Check service status
gcloud run services describe gpt-oss-20b-api --region=us-central1

# View logs
gcloud logs tail --service=gpt-oss-20b-api

# Test health endpoints
curl -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  https://your-service-url/ready

# Check GPU availability
curl -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  https://your-service-url/ready | jq '.gpu'
```

## Development

### Project Structure
```
.
├── app/
│   ├── main.py          # FastAPI application
│   ├── config.py        # Configuration management
│   ├── models.py        # Pydantic models (OpenAI compatible)
│   ├── llm_engine.py    # vLLM engine management
│   ├── streaming.py     # SSE streaming handlers
│   └── health.py        # Health check endpoints
├── Dockerfile           # GPU-enabled container
├── docker-compose.yml   # Local development
├── cloudbuild.yaml      # Cloud Build configuration
├── deploy.sh           # Deployment script
├── requirements.txt    # Python dependencies
└── .env.example       # Environment template
```

### Adding Features
1. Update Pydantic models in `app/models.py`
2. Add endpoint handlers in `app/main.py`
3. Update streaming logic in `app/streaming.py` if needed
4. Test locally with docker-compose
5. Deploy and test on Cloud Run

## Security

- **Authentication**: IAM-based authentication via Cloud Run
- **Network**: Private Google network for backend services
- **Secrets**: Use Google Secret Manager for sensitive configuration
- **Container**: Minimal attack surface with distroless base images
- **API**: Input validation via Pydantic models

## License

This project is licensed under the Apache License 2.0. The GPT-OSS-20B model is also Apache 2.0 licensed for commercial use.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Cloud Run logs for error details
3. Test locally with docker-compose for debugging
4. Verify GPU availability and model loading via `/ready` endpoint
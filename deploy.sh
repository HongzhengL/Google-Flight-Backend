#!/bin/bash

# GPT-OSS-20B Cloud Run Deployment Script

set -e  # Exit on error

# Configuration variables
PROJECT_ID=${PROJECT_ID:-"your-gcp-project-id"}
SERVICE_NAME=${SERVICE_NAME:-"gpt-oss-20b-api"}
REGION=${REGION:-"us-central1"}
ARTIFACT_REGISTRY_REPO=${ARTIFACT_REGISTRY_REPO:-"cloud-run-source-deploy"}
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${SERVICE_NAME}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting GPT-OSS-20B deployment to Cloud Run...${NC}"

# Validate required environment variables
if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "your-gcp-project-id" ]]; then
    echo -e "${RED}Error: PROJECT_ID must be set${NC}"
    echo "Usage: PROJECT_ID=your-project-id ./deploy.sh"
    exit 1
fi

# Set the project
echo -e "${YELLOW}Setting GCP project: $PROJECT_ID${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs if not already enabled
echo -e "${YELLOW}Ensuring required APIs are enabled...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Create Artifact Registry repository if it doesn't exist
echo -e "${YELLOW}Creating Artifact Registry repository if needed...${NC}"
gcloud artifacts repositories create $ARTIFACT_REGISTRY_REPO \
    --repository-format=docker \
    --location=$REGION \
    --quiet || echo "Repository already exists"

# Configure Docker authentication
echo -e "${YELLOW}Configuring Docker authentication...${NC}"
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build and push the container image
echo -e "${YELLOW}Building container image...${NC}"
docker build -t $IMAGE_NAME:latest .

echo -e "${YELLOW}Pushing container image to Artifact Registry...${NC}"
docker push $IMAGE_NAME:latest

# Deploy to Cloud Run with GPU support
echo -e "${YELLOW}Deploying to Cloud Run with GPU...${NC}"
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME:latest \
    --platform managed \
    --region $REGION \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --memory 32Gi \
    --cpu 8 \
    --timeout 1200 \
    --concurrency 4 \
    --max-instances 1 \
    --min-instances 0 \
    --port 8080 \
    --no-allow-unauthenticated \
    --execution-environment gen2 \
    --no-gpu-zonal-redundancy \
    --set-env-vars "MAX_NUM_SEQS=4,GPU_MEMORY_UTILIZATION=0.90,MODEL_NAME=openai/gpt-oss-20b,LOG_LEVEL=INFO" \
    --no-startup-probe \
    --quiet

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform=managed --region=$REGION --format='value(status.url)')

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "${GREEN}Service URL: $SERVICE_URL${NC}"
echo ""
echo -e "${YELLOW}Health check:${NC} curl -H \"Authorization: Bearer \$(gcloud auth print-access-token)\" $SERVICE_URL/health"
echo -e "${YELLOW}Readiness check:${NC} curl -H \"Authorization: Bearer \$(gcloud auth print-access-token)\" $SERVICE_URL/ready"
echo -e "${YELLOW}API Documentation:${NC} $SERVICE_URL/docs"
echo ""
echo -e "${YELLOW}Example API call:${NC}"
echo "curl -X POST \"$SERVICE_URL/v1/chat/completions\" \\"
echo "  -H \"Authorization: Bearer \$(gcloud auth print-access-token)\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{"
echo "    \"model\": \"gpt-oss-20b\","
echo "    \"messages\": ["
echo "      {\"role\": \"user\", \"content\": \"Hello, how are you?\"}"
echo "    ],"
echo "    \"stream\": true"
echo "  }'"
#!/bin/bash

# Docker XAI Integration Test Script
# Tests that XAI modules work properly in Docker containers

set -e

echo "🐳 DOCKER XAI INTEGRATION TEST"
echo "==============================="

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m' # No Color

# Function to print colored output
print_status() {
    case $1 in
        "INFO") echo -e "${BLUE}ℹ️  $2${NC}" ;;
        "SUCCESS") echo -e "${GREEN}✅ $2${NC}" ;;
        "WARNING") echo -e "${YELLOW}⚠️  $2${NC}" ;;
        "ERROR") echo -e "${RED}❌ $2${NC}" ;;
    esac
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_status "ERROR" "Docker is not running. Please start Docker first."
    exit 1
fi

print_status "SUCCESS" "Docker is running"

# Build the Docker image
print_status "INFO" "Building Docker image..."
docker-compose -f docker/docker-compose.yml build health-xai-app

if [ $? -eq 0 ]; then
    print_status "SUCCESS" "Docker image built successfully"
else
    print_status "ERROR" "Docker image build failed"
    exit 1
fi

# Run XAI compatibility tests
print_status "INFO" "Running XAI compatibility tests..."
docker-compose -f docker/docker-compose.yml --profile test run --rm xai-test

if [ $? -eq 0 ]; then
    print_status "SUCCESS" "XAI compatibility tests passed"
else
    print_status "ERROR" "XAI compatibility tests failed"
    exit 1
fi

# Test Jupyter service
print_status "INFO" "Testing Jupyter service startup..."
docker-compose -f docker/docker-compose.yml up -d jupyter

# Wait for Jupyter to start
sleep 10

# Check if Jupyter is accessible
if curl -s http://localhost:8889 > /dev/null; then
    print_status "SUCCESS" "Jupyter service is accessible"
else
    print_status "WARNING" "Jupyter service may not be fully ready"
fi

# Stop Jupyter service
docker-compose -f docker/docker-compose.yml down

print_status "SUCCESS" "Docker XAI integration test completed!"

echo ""
echo "📋 Next Steps:"
echo "1. Run: docker-compose -f docker/docker-compose.yml up health-xai-app (for Gradio app)"
echo "2. Run: docker-compose -f docker/docker-compose.yml up jupyter (for Jupyter development)"
echo "3. Run: docker-compose -f docker/docker-compose.yml --profile test run xai-test (for XAI tests)"

echo ""
echo "🌐 Access URLs:"
echo "• Gradio App: http://localhost:7860"
echo "• Jupyter Lab: http://localhost:8889"
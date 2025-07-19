#!/bin/bash

# Test script for the model serving API

set -e

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

API_URL=${API_URL:-"http://localhost:8000"}

echo -e "${GREEN}Testing Iris Classification API at $API_URL${NC}"

# Function to test endpoint
test_endpoint() {
    local endpoint=$1
    local method=${2:-GET}
    local data=${3:-""}
    
    echo -e "${YELLOW}Testing $method $endpoint${NC}"
    
    if [ "$method" = "POST" ] && [ -n "$data" ]; then
        curl -s -X POST "$API_URL$endpoint" \
             -H "Content-Type: application/json" \
             -d "$data" | jq .
    else
        curl -s "$API_URL$endpoint" | jq .
    fi
    
    echo ""
}

# Test root endpoint
test_endpoint "/"

# Test health check
test_endpoint "/health"

# Test model info
test_endpoint "/model/info"

# Test single prediction
echo -e "${YELLOW}Testing single prediction (Setosa)${NC}"
SETOSA_DATA='{"features": [5.1, 3.5, 1.4, 0.2]}'
test_endpoint "/predict" "POST" "$SETOSA_DATA"

echo -e "${YELLOW}Testing single prediction (Versicolor)${NC}"
VERSICOLOR_DATA='{"features": [6.2, 2.9, 4.3, 1.3]}'
test_endpoint "/predict" "POST" "$VERSICOLOR_DATA"

echo -e "${YELLOW}Testing single prediction (Virginica)${NC}"
VIRGINICA_DATA='{"features": [7.3, 2.9, 6.3, 1.8]}'
test_endpoint "/predict" "POST" "$VIRGINICA_DATA"

# Test batch prediction
echo -e "${YELLOW}Testing batch prediction${NC}"
BATCH_DATA='{
  "features": [
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 2.9, 4.3, 1.3],
    [7.3, 2.9, 6.3, 1.8],
    [4.6, 3.1, 1.5, 0.2],
    [5.9, 3.0, 5.1, 1.8]
  ]
}'
test_endpoint "/predict/batch" "POST" "$BATCH_DATA"

# Test error handling
echo -e "${YELLOW}Testing error handling (invalid features)${NC}"
INVALID_DATA='{"features": [1, 2, 3]}'
curl -s -X POST "$API_URL/predict" \
     -H "Content-Type: application/json" \
     -d "$INVALID_DATA" | jq .

echo ""
echo -e "${GREEN}API testing completed!${NC}"
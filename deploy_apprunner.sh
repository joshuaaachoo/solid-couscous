#!/bin/bash
set -e

# RiftRewind - AWS App Runner Deployment Script
# This deploys the Streamlit app with DynamoDB caching

# Configuration
APP_NAME="riftrewind"
AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPO_NAME="riftrewind-app"
SERVICE_NAME="riftrewind-streamlit"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 RiftRewind - AWS App Runner Deployment${NC}"
echo "========================================"

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI not found. Please install it first.${NC}"
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install it first.${NC}"
    exit 1
fi

# Get AWS account ID
echo -e "${BLUE}📋 Getting AWS account info...${NC}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "Account ID: $AWS_ACCOUNT_ID"

# Create ECR repository if it doesn't exist
echo -e "${BLUE}📦 Setting up ECR repository...${NC}"
if ! aws ecr describe-repositories --repository-names $ECR_REPO_NAME --region $AWS_REGION 2>/dev/null; then
    echo "Creating ECR repository: $ECR_REPO_NAME"
    aws ecr create-repository \
        --repository-name $ECR_REPO_NAME \
        --region $AWS_REGION \
        --image-scanning-configuration scanOnPush=true
    echo -e "${GREEN}✅ ECR repository created${NC}"
else
    echo -e "${GREEN}✅ ECR repository exists${NC}"
fi

# Get ECR login
echo -e "${BLUE}🔐 Logging into ECR...${NC}"
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build Docker image
echo -e "${BLUE}🔨 Building Docker image...${NC}"
IMAGE_TAG="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest"
docker build -t $ECR_REPO_NAME:latest .
docker tag $ECR_REPO_NAME:latest $IMAGE_TAG
echo -e "${GREEN}✅ Docker image built${NC}"

# Push to ECR
echo -e "${BLUE}⬆️  Pushing image to ECR...${NC}"
docker push $IMAGE_TAG
echo -e "${GREEN}✅ Image pushed to ECR${NC}"

# Check if App Runner service exists
echo -e "${BLUE}🔍 Checking for existing App Runner service...${NC}"
if aws apprunner list-services --region $AWS_REGION --query "ServiceSummaryList[?ServiceName=='$SERVICE_NAME']" --output text | grep -q "$SERVICE_NAME"; then
    echo -e "${YELLOW}⚠️  Service exists. Updating...${NC}"
    
    # Get service ARN
    SERVICE_ARN=$(aws apprunner list-services --region $AWS_REGION \
        --query "ServiceSummaryList[?ServiceName=='$SERVICE_NAME'].ServiceArn" \
        --output text)
    
    # Update service
    aws apprunner update-service \
        --service-arn $SERVICE_ARN \
        --region $AWS_REGION \
        --source-configuration "ImageRepository={ImageIdentifier=$IMAGE_TAG,ImageRepositoryType=ECR,ImageConfiguration={Port=8501}}" \
        --output json > /dev/null
    
    echo -e "${GREEN}✅ Service updated${NC}"
else
    echo -e "${BLUE}🆕 Creating new App Runner service...${NC}"
    
    # Create IAM role for App Runner if it doesn't exist
    ROLE_NAME="AppRunnerECRAccessRole"
    if ! aws iam get-role --role-name $ROLE_NAME 2>/dev/null; then
        echo "Creating IAM role for App Runner..."
        
        # Create trust policy
        cat > /tmp/trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "build.apprunner.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
        
        aws iam create-role \
            --role-name $ROLE_NAME \
            --assume-role-policy-document file:///tmp/trust-policy.json
        
        aws iam attach-role-policy \
            --role-name $ROLE_NAME \
            --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
        
        echo "Waiting for role to propagate..."
        sleep 10
    fi
    
    ROLE_ARN="arn:aws:iam::$AWS_ACCOUNT_ID:role/$ROLE_NAME"
    
    # Create service
    aws apprunner create-service \
        --service-name $SERVICE_NAME \
        --region $AWS_REGION \
        --source-configuration "{
            \"ImageRepository\": {
                \"ImageIdentifier\": \"$IMAGE_TAG\",
                \"ImageRepositoryType\": \"ECR\",
                \"ImageConfiguration\": {
                    \"Port\": \"8501\",
                    \"RuntimeEnvironmentVariables\": {
                        \"AWS_REGION\": \"$AWS_REGION\",
                        \"RIOT_API_KEY\": \"$RIOT_API_KEY\",
                        \"MATCHES_TABLE\": \"riftrewind-matches\",
                        \"SEARCHES_TABLE\": \"riftrewind-searches\"
                    }
                }
            },
            \"AuthenticationConfiguration\": {
                \"AccessRoleArn\": \"$ROLE_ARN\"
            },
            \"AutoDeploymentsEnabled\": false
        }" \
        --instance-configuration "{
            \"Cpu\": \"1 vCPU\",
            \"Memory\": \"2 GB\"
        }" \
        --health-check-configuration "{
            \"Protocol\": \"HTTP\",
            \"Path\": \"/_stcore/health\",
            \"Interval\": 10,
            \"Timeout\": 5,
            \"HealthyThreshold\": 1,
            \"UnhealthyThreshold\": 5
        }" \
        --output json > /tmp/apprunner-service.json
    
    echo -e "${GREEN}✅ Service created${NC}"
fi

# Get service URL
echo -e "${BLUE}⏳ Waiting for service to be ready...${NC}"
sleep 10

SERVICE_ARN=$(aws apprunner list-services --region $AWS_REGION \
    --query "ServiceSummaryList[?ServiceName=='$SERVICE_NAME'].ServiceArn" \
    --output text)

SERVICE_URL=$(aws apprunner describe-service \
    --service-arn $SERVICE_ARN \
    --region $AWS_REGION \
    --query "Service.ServiceUrl" \
    --output text)

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}📱 Service URL:${NC}"
echo "   https://$SERVICE_URL"
echo ""
echo -e "${BLUE}📊 View in AWS Console:${NC}"
echo "   https://console.aws.amazon.com/apprunner/home?region=$AWS_REGION#/services/$SERVICE_NAME"
echo ""
echo -e "${YELLOW}⏰ Note: It may take 2-3 minutes for the service to be fully operational${NC}"
echo ""

# Check environment variables
echo -e "${BLUE}🔧 Service Configuration:${NC}"
echo "   - DynamoDB: riftrewind-matches, riftrewind-searches"
echo "   - Region: $AWS_REGION"
echo "   - Instance: 1 vCPU, 2 GB RAM"
echo ""
echo -e "${GREEN}🎉 Your RiftRewind app is now running on AWS!${NC}"

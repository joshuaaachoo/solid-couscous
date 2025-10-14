#!/bin/bash
set -e

# RiftRewind - AWS App Runner Deployment (GitHub Source)
# This deploys directly from GitHub without requiring Docker locally

# Configuration
SERVICE_NAME="riftrewind-streamlit"
AWS_REGION="${AWS_REGION:-us-east-1}"
GITHUB_REPO="joshuaaachoo/solid-couscous"
GITHUB_BRANCH="${GITHUB_BRANCH:-main}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 RiftRewind - AWS App Runner Deployment${NC}"
echo "========================================"

# Check for required environment variables
if [ -z "$RIOT_API_KEY" ]; then
    echo -e "${RED}❌ RIOT_API_KEY not set!${NC}"
    echo "Please set it: export RIOT_API_KEY=your_key_here"
    exit 1
fi

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI not found. Please install it first.${NC}"
    exit 1
fi

# Get AWS account ID
echo -e "${BLUE}📋 Getting AWS account info...${NC}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "Account ID: $AWS_ACCOUNT_ID"
echo "Region: $AWS_REGION"

# Check if App Runner service exists
echo -e "${BLUE}🔍 Checking for existing App Runner service...${NC}"
SERVICE_EXISTS=$(aws apprunner list-services --region $AWS_REGION \
    --query "ServiceSummaryList[?ServiceName=='$SERVICE_NAME'].ServiceArn" \
    --output text 2>/dev/null || echo "")

if [ -n "$SERVICE_EXISTS" ]; then
    echo -e "${YELLOW}⚠️  Service '$SERVICE_NAME' already exists${NC}"
    echo "Service ARN: $SERVICE_EXISTS"
    echo ""
    echo "Options:"
    echo "  1. Update existing service"
    echo "  2. Delete and recreate"
    echo "  3. Cancel"
    read -p "Choose (1-3): " choice
    
    case $choice in
        1)
            echo -e "${BLUE}🔄 Updating service...${NC}"
            # For GitHub source, we need to trigger a new deployment
            aws apprunner start-deployment \
                --service-arn $SERVICE_EXISTS \
                --region $AWS_REGION
            echo -e "${GREEN}✅ Deployment triggered${NC}"
            ;;
        2)
            echo -e "${YELLOW}🗑️  Deleting existing service...${NC}"
            aws apprunner delete-service \
                --service-arn $SERVICE_EXISTS \
                --region $AWS_REGION
            echo "Waiting for service to be deleted..."
            sleep 30
            SERVICE_EXISTS=""
            ;;
        *)
            echo "Cancelled"
            exit 0
            ;;
    esac
fi

# Create service if it doesn't exist
if [ -z "$SERVICE_EXISTS" ]; then
    echo -e "${BLUE}🆕 Creating new App Runner service from GitHub...${NC}"
    
    # Check for GitHub connection
    echo -e "${BLUE}🔗 Checking GitHub connection...${NC}"
    GITHUB_CONNECTION=$(aws apprunner list-connections --region $AWS_REGION \
        --query "ConnectionSummaryList[?Status=='AVAILABLE'].ConnectionArn | [0]" \
        --output text 2>/dev/null || echo "")
    
    if [ -z "$GITHUB_CONNECTION" ] || [ "$GITHUB_CONNECTION" == "None" ]; then
        echo -e "${RED}❌ No GitHub connection found!${NC}"
        echo ""
        echo "You need to create a GitHub connection in App Runner:"
        echo "1. Go to: https://console.aws.amazon.com/apprunner/home?region=$AWS_REGION#/connections"
        echo "2. Click 'Add connection'"
        echo "3. Select 'GitHub' and authorize"
        echo "4. Re-run this script"
        echo ""
        exit 1
    fi
    
    echo -e "${GREEN}✅ Using GitHub connection: $GITHUB_CONNECTION${NC}"
    
    # Create instance role for DynamoDB access
    ROLE_NAME="RiftRewindAppRunnerInstanceRole"
    echo -e "${BLUE}🔐 Setting up IAM role...${NC}"
    
    if ! aws iam get-role --role-name $ROLE_NAME 2>/dev/null; then
        echo "Creating instance role..."
        
        cat > /tmp/instance-trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "tasks.apprunner.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
        
        aws iam create-role \
            --role-name $ROLE_NAME \
            --assume-role-policy-document file:///tmp/instance-trust-policy.json
        
        # Attach DynamoDB policy
        cat > /tmp/dynamodb-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:Query",
        "dynamodb:Scan",
        "dynamodb:UpdateItem",
        "dynamodb:DeleteItem",
        "dynamodb:DescribeTable"
      ],
      "Resource": [
        "arn:aws:dynamodb:$AWS_REGION:$AWS_ACCOUNT_ID:table/riftrewind-*"
      ]
    }
  ]
}
EOF
        
        aws iam put-role-policy \
            --role-name $ROLE_NAME \
            --policy-name DynamoDBAccess \
            --policy-document file:///tmp/dynamodb-policy.json
        
        echo "Waiting for role to propagate..."
        sleep 10
    fi
    
    INSTANCE_ROLE_ARN="arn:aws:iam::$AWS_ACCOUNT_ID:role/$ROLE_NAME"
    
    # Create service configuration
    cat > /tmp/apprunner-config.json <<EOF
{
  "ServiceName": "$SERVICE_NAME",
  "SourceConfiguration": {
    "AuthenticationConfiguration": {
      "ConnectionArn": "$GITHUB_CONNECTION"
    },
    "AutoDeploymentsEnabled": true,
    "CodeRepository": {
      "RepositoryUrl": "https://github.com/$GITHUB_REPO",
      "SourceCodeVersion": {
        "Type": "BRANCH",
        "Value": "$GITHUB_BRANCH"
      },
      "CodeConfiguration": {
        "ConfigurationSource": "API",
        "CodeConfigurationValues": {
          "Runtime": "PYTHON_3",
          "BuildCommand": "pip install -r requirements.txt",
          "StartCommand": "streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true",
          "Port": "8501",
          "RuntimeEnvironmentVariables": {
            "AWS_REGION": "$AWS_REGION",
            "RIOT_API_KEY": "$RIOT_API_KEY",
            "MATCHES_TABLE": "riftrewind-matches",
            "SEARCHES_TABLE": "riftrewind-searches"
          }
        }
      }
    }
  },
  "InstanceConfiguration": {
    "Cpu": "1 vCPU",
    "Memory": "2 GB",
    "InstanceRoleArn": "$INSTANCE_ROLE_ARN"
  },
  "HealthCheckConfiguration": {
    "Protocol": "HTTP",
    "Path": "/_stcore/health",
    "Interval": 10,
    "Timeout": 5,
    "HealthyThreshold": 1,
    "UnhealthyThreshold": 5
  }
}
EOF
    
    echo -e "${BLUE}🚀 Creating App Runner service...${NC}"
    aws apprunner create-service \
        --cli-input-json file:///tmp/apprunner-config.json \
        --region $AWS_REGION \
        --output json > /tmp/apprunner-service.json
    
    SERVICE_EXISTS=$(cat /tmp/apprunner-service.json | grep -o '"ServiceArn": "[^"]*' | cut -d'"' -f4)
    echo -e "${GREEN}✅ Service created!${NC}"
fi

# Wait for service to be ready
echo -e "${BLUE}⏳ Waiting for service to be operational...${NC}"
echo "(This can take 3-5 minutes for the first deployment)"

for i in {1..60}; do
    STATUS=$(aws apprunner describe-service \
        --service-arn $SERVICE_EXISTS \
        --region $AWS_REGION \
        --query "Service.Status" \
        --output text)
    
    if [ "$STATUS" == "RUNNING" ]; then
        echo -e "\n${GREEN}✅ Service is RUNNING!${NC}"
        break
    elif [ "$STATUS" == "CREATE_FAILED" ] || [ "$STATUS" == "UPDATE_FAILED" ]; then
        echo -e "\n${RED}❌ Deployment failed!${NC}"
        aws apprunner describe-service \
            --service-arn $SERVICE_EXISTS \
            --region $AWS_REGION \
            --query "Service.Status" \
            --output text
        exit 1
    else
        echo -n "."
        sleep 5
    fi
done

# Get service URL
SERVICE_URL=$(aws apprunner describe-service \
    --service-arn $SERVICE_EXISTS \
    --region $AWS_REGION \
    --query "Service.ServiceUrl" \
    --output text)

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}📱 Your App:${NC}"
echo "   https://$SERVICE_URL"
echo ""
echo -e "${BLUE}📊 AWS Console:${NC}"
echo "   https://console.aws.amazon.com/apprunner/home?region=$AWS_REGION#/services/$SERVICE_NAME"
echo ""
echo -e "${BLUE}🔧 Features Enabled:${NC}"
echo "   ✅ DynamoDB Caching (matches + searches)"
echo "   ✅ Streamlit UI"
echo "   ✅ Auto-deployment from GitHub"
echo "   ✅ Health checks"
echo ""
echo -e "${YELLOW}📝 Next Steps:${NC}"
echo "   1. Test the app at the URL above"
echo "   2. Add Amazon Bedrock for AI insights"
echo "   3. Add Amazon Comprehend for toxicity detection"
echo "   4. Create CloudWatch dashboard"
echo ""
echo -e "${GREEN}🎉 RiftRewind is live on AWS!${NC}"

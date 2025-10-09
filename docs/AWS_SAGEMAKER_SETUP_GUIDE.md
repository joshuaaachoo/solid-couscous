# 🔧 Step-by-Step Guide: Adding AWS SageMaker Permissions
*For hackathon-user in AWS Account 407275151589*

## 🎯 **Complete Setup Guide for Option 2**

### **Step 1: Access AWS IAM Console**

1. **Go to AWS Console**: https://aws.amazon.com/console/
2. **Sign in** with your AWS account credentials
3. **Navigate to IAM**:
   - Search "IAM" in the services search bar
   - Click on "IAM" (Identity and Access Management)

### **Step 2: Find Your User**

1. **Click "Users"** in the left sidebar
2. **Search for your user**: `hackathon-user`
3. **Click on the username** to open user details

### **Step 3: Create the SageMaker Policy**

#### **Option A: Create Custom Policy (Recommended)**

1. **Go to Policies** (left sidebar)
2. **Click "Create policy"**
3. **Click the "JSON" tab**
4. **Paste this policy**:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "RiftRewindSageMakerAccess",
            "Effect": "Allow",
            "Action": [
                "sagemaker:DescribeEndpoint",
                "sagemaker:DescribeEndpointConfig", 
                "sagemaker:DescribeModel",
                "sagemaker:CreateModel",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:CreateEndpoint",
                "sagemaker:DeleteEndpoint",
                "sagemaker:DeleteEndpointConfig",
                "sagemaker:DeleteModel",
                "sagemaker:InvokeEndpoint",
                "sagemaker:ListEndpoints",
                "sagemaker:ListModels"
            ],
            "Resource": "*"
        },
        {
            "Sid": "RiftRewindIAMPassRole",
            "Effect": "Allow", 
            "Action": "iam:PassRole",
            "Resource": "arn:aws:iam::407275151589:role/service-role/*SageMaker*"
        },
        {
            "Sid": "RiftRewindS3Access",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::*sagemaker*",
                "arn:aws:s3:::*sagemaker*/*",
                "arn:aws:s3:::*riftrewind*",
                "arn:aws:s3:::*riftrewind*/*"
            ]
        }
    ]
}
```

5. **Click "Next"**
6. **Name the policy**: `RiftRewindSageMakerPolicy`
7. **Add description**: `Custom policy for RiftRewind VOD analysis SageMaker access`
8. **Click "Create policy"**

#### **Option B: Use AWS Managed Policy (Easier)**

Alternative: Use AWS's pre-built policy `AmazonSageMakerFullAccess`
- **Pros**: Easier setup, comprehensive permissions
- **Cons**: Broader permissions than needed

### **Step 4: Attach Policy to Your User**

1. **Go back to Users** → **hackathon-user**
2. **Click "Add permissions"** button
3. **Select "Attach policies directly"**
4. **Search for your policy**:
   - If you created custom: `RiftRewindSageMakerPolicy`  
   - If using managed: `AmazonSageMakerFullAccess`
5. **Check the box** next to the policy
6. **Click "Next"** → **"Add permissions"**

### **Step 5: Set Up AWS Credentials on Your Computer**

**You need your AWS Access Keys for hackathon-user:**

1. **Go back to Users** → **hackathon-user** 
2. **Click "Security credentials" tab**
3. **Click "Create access key"** (if you don't have one)
4. **Copy the Access Key ID and Secret Access Key**

**Configure credentials using AWS CLI:**

```bash
# Use the AWS CLI in your project
cd /Users/joshs/RiftRewindClean
.venv/bin/aws configure
```

**When prompted, enter:**
- **AWS Access Key ID**: `[Your hackathon-user access key]`
- **AWS Secret Access Key**: `[Your hackathon-user secret key]`
- **Default region**: `us-east-1`
- **Default output format**: `json`

### **Step 6: Verify Setup**

1. **Check AWS credentials work:**

```bash
.venv/bin/aws sts get-caller-identity
```

2. **Test in your app**:

```bash
cd /Users/joshs/RiftRewindClean
streamlit run app.py
```

## 🔍 **Troubleshooting Common Issues**

### **Issue 1: "No SageMaker Service Role"**
**Error**: `Cannot assume role` or `PassRole` errors

**Solution**: Create a SageMaker execution role:
1. **IAM Console** → **Roles** → **Create role**
2. **Select "SageMaker"** as trusted entity
3. **Attach policy**: `AmazonSageMakerExecutionRole`
4. **Name**: `RiftRewind-SageMaker-ExecutionRole`

### **Issue 2: "Region Not Available"**
**Error**: SageMaker not available in region

**Solution**: Change your AWS region:
1. **Top-right corner** of AWS Console
2. **Select a SageMaker-supported region**: 
   - `us-east-1` (N. Virginia) ✅ Recommended
   - `us-west-2` (Oregon) ✅
   - `eu-west-1` (Ireland) ✅

### **Issue 3: "Account Limits"**
**Error**: Resource limit exceeded

**Solution**: Request limit increase:
1. **AWS Support Center** → **Create case**
2. **Service limit increase** → **SageMaker**

## 🎯 **Minimal Permissions (Security-Focused)**

If you prefer minimal permissions, use this reduced policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:DescribeEndpoint",
                "sagemaker:InvokeEndpoint"
            ],
            "Resource": [
                "arn:aws:sagemaker:us-east-1:407275151589:endpoint/riftrewind-*"
            ]
        }
    ]
}
```

This only allows:
- ✅ **Checking endpoint status** (fixes the permission error)
- ✅ **Running inference** (for actual analysis)
- ❌ **No model deployment** (more secure)

## ✅ **Success Verification**

After completing the setup, your RiftRewind app will show:
- 🔒 **No more permission errors**
- ✅ **Green status indicators** for SageMaker models
- 🚀 **Ready to deploy** buttons active
- 📊 **Full model management** capabilities

## 🚨 **Important Notes**

1. **Billing**: SageMaker instances cost money when running
2. **Security**: Only grant minimum required permissions
3. **Region**: Ensure consistent region across services
4. **Cleanup**: Delete endpoints when not in use to save costs

---
**After completing these steps, your RiftRewind Pro will have full SageMaker capabilities!** 🎯
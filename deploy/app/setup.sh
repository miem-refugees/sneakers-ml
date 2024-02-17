#!/bin/sh

mkdir -p ~/.aws

cat <<EOF > ~/.aws/config
[default]
region = ru-central1
EOF
echo "AWS config created successfully: ~/.aws/config"


if [ -f ~/.aws/credentials ]; then
  echo "AWS credentials already exists, skipping..."
  exit 0
fi

if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
  echo "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are not set."
  exit 1
fi

cat <<EOF > ~/.aws/credentials
[default]
aws_access_key_id = $AWS_ACCESS_KEY_ID
aws_secret_access_key = $AWS_SECRET_ACCESS_KEY
EOF
echo "AWS credentials created successfully: ~/.aws/credentials"

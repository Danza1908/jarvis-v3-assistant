# JARVIS V3 Setup Guide

This guide will help you set up your OpenAI API key for JARVIS V3.

## 🔑 Getting Your OpenAI API Key

1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in to your account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-`)

## 🛠️ Option 1: Using Streamlit Secrets (Recommended)

1. Edit `.streamlit/secrets.toml`
2. Replace `your-openai-api-key-here` with your actual API key:
   ```toml
   OPENAI_API_KEY = "sk-your-actual-key-here"
   ```

## 🛠️ Option 2: Using Environment Variables

1. Copy `.env.example` to `.env`:
   ```bash
   copy .env.example .env
   ```
2. Edit `.env` and add your API key
3. Run Docker with environment file:
   ```bash
   docker run -d -p 8501:8501 --env-file .env --name jarvis-v3-container jarvis-v3
   ```

## 🛠️ Option 3: Direct Input in App

1. Run the application
2. Enter your API key in the sidebar
3. The key will be stored for the session

## 🚀 Quick Start Commands

### Build and Run with Secrets:
```bash
# Stop existing container
docker stop jarvis-v3-container
docker rm jarvis-v3-container

# Rebuild with new configuration
docker build -t jarvis-v3 .

# Run container
docker run -d -p 8501:8501 --name jarvis-v3-container jarvis-v3
```

### Build and Run with Environment File:
```bash
# After setting up .env file
docker run -d -p 8501:8501 --env-file .env --name jarvis-v3-container jarvis-v3
```

## 📱 Access Your App

Open your browser and go to: http://localhost:8501

## 🔧 Troubleshooting

- Make sure your API key starts with `sk-`
- Check that you have credit available in your OpenAI account
- Verify the API key is correctly placed in the configuration file
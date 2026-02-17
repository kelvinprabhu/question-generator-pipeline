# 🐳 Docker Usage Guide

This guide explains how to build and run the Multi-Intent Question Generator using Docker.

## 1. Build the Image

You must provide your API keys as build arguments:

```bash
docker build \
  --build-arg GROQ_API_KEYS="your_groq_key" \
  --build-arg GEMINI_API_KEYS="your_gemini_key" \
  --build-arg HF_API_KEYS="your_hf_key" \
  --build-arg OPENROUTER_API_KEYS="your_or_key" \
  --build-arg MONGO_URI="mongodb://host.docker.internal:27017/" \
  --build-arg MONGO_DB_NAME="questions" \
  -t question-generator .
```

> **Note:** Use `host.docker.internal` for `MONGO_URI` to connect to MongoDB running on your host machine.

## 2. Run the Container

Once built, run the container just like the CLI script:

### Basic Run (50 Questions)
```bash
docker run --rm question-generator --total 50
```

### With Specific Persona
```bash
docker run --rm question-generator \
  --total 100 \
  --persona "A progressive soya bean farmer from Indore using precision agriculture"
```

### Dry Run (Test Mode)
```bash
docker run --rm question-generator --dry-run
```

### Advanced Usage (Batch Size & Strategy)
Run with a smaller batch size and coverage-based strategy:
```bash
docker run --rm question-generator \
  --total 500 \
  --batch-size 5 \
  --strategy coverage_based
```

## 3. Using Pre-built Image (from Docker Hub)

If you have configured the CI/CD pipeline, pull and run directly:

```bash
docker pull your-username/multi-intent-question-generator:latest

docker run --rm \
  -e GROQ_API_KEYS="override_at_runtime" \
  your-username/multi-intent-question-generator:latest --total 50
```
*(If keys were baked in at build time, you don't need `-e`, but it's safer to override/pass them at runtime if they weren't)*

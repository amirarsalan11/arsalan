# Placeholder Dockerfile for the AI Engine / GPU worker service.
#
# Milestone 1 scope: builds successfully and can boot a minimal Python
# process. No model weights, CUDA base image, or GPU runtime are
# configured yet — those are introduced in Milestone 7 (SegFormer
# Integration) and the Async Worker System milestone.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-c", "print('AI Engine placeholder container - no pipeline implemented yet')"]

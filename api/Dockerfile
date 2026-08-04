# Base image: official Python, "slim" = smaller size (fewer preinstalled OS packages)
FROM python:3.12-slim

# Set working directory inside the container — everything from here on happens in /app
WORKDIR /app

# Copy ONLY requirements.txt first (not the whole repo yet) — this is a deliberate caching trick,
# explained below the code block
COPY requirements.txt .

# Install dependencies. --no-cache-dir keeps the image smaller (pip doesn't keep its download cache)
RUN pip install --no-cache-dir -r requirements.txt

# NOW copy the actual application code
COPY api/ ./api/
COPY ingestion/ ./ingestion/
COPY rag/ ./rag/
COPY vectorstore/ ./vectorstore/

# Tell Docker (informational only, doesn't actually open the port) that this container listens on 8000
EXPOSE 8000

# The command that runs when the container starts
CMD ["python", "api/main.py"]

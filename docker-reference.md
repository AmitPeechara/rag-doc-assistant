# Docker Quick Reference — rag-doc-assistant

## Concepts to remember
- **Image** — immutable snapshot of your app + OS + dependencies. Built once, run anywhere.
- **Container** — a running instance of an image. One image → many containers.
- **Layer cache** — each Dockerfile instruction is cached. Change in line N invalidates N onwards.

---

## 1. Build images
Always run from the **repo root** (the `.` at the end is the build context — Docker needs to see the whole repo to COPY files from it).

```bash
# API image
docker build -t rag-api:v1 -f api/Dockerfile .

# UI image
docker build -t rag-ui:v1 -f ui/Dockerfile .
```

`-t` = tag (name:version)
`-f` = which Dockerfile to use (needed when it's not in the current directory)
`.` = build context (use the current directory, i.e. repo root)

---

## 2. Check what images exist
```bash
docker images
```

---

## 3. Create the network (once — only needed if it doesn't exist yet)
```bash
docker network create rag-net
```
This lets containers reach each other by name instead of localhost.

---

## 4. Run containers

**API:**
```bash
docker run -d \
  --name rag-api \
  --network rag-net \
  -p 8000:8000 \
  --env-file .env \
  rag-api:v1
```

**UI:**
```bash
docker run -d \
  --name rag-ui \
  --network rag-net \
  -p 8501:8501 \
  -e API_BASE_URL=http://rag-api:8000 \
  rag-ui:v1
```

Flag reference:
| Flag | What it does |
|---|---|
| `-d` | Run in background (detached), gives terminal back |
| `--name` | Give the container a fixed name (also acts as hostname on the network) |
| `--network` | Attach to a Docker network so containers can reach each other by name |
| `-p host:container` | Publish a port — makes the container reachable from your laptop |
| `--env-file .env` | Inject all variables from a .env file into the container |
| `-e KEY=value` | Inject a single environment variable |

---

## 5. Check running containers
```bash
docker ps        # running containers only
docker ps -a     # all containers including stopped ones
```

---

## 6. View logs (if something's wrong)
```bash
docker logs rag-api
docker logs rag-ui
docker logs -f rag-api   # -f = follow (live tail, like tail -f)
```

---

## 7. Stop and remove containers
```bash
docker stop rag-api rag-ui
docker rm rag-api rag-ui
```
Stop halts the container. rm removes it (must remove before you can reuse the name).

---

## 8. Remove images (only if you want to clean up)
```bash
docker rmi rag-api:v1 rag-ui:v1
```
Will fail if a container (even stopped) still references the image — remove the container first.

---

## Full restart from scratch (most common flow during development)
```bash
# 1. Stop and remove old containers
docker stop rag-api rag-ui
docker rm rag-api rag-ui

# 2. Rebuild images (bump version tag if you made changes)
docker build -t rag-api:v2 -f api/Dockerfile .
docker build -t rag-ui:v2 -f ui/Dockerfile .

# 3. Run again
docker run -d --name rag-api --network rag-net -p 8000:8000 --env-file .env rag-api:v2
docker run -d --name rag-ui --network rag-net -p 8501:8501 -e API_BASE_URL=http://rag-api:8000 rag-ui:v2
```

---

## Access points
| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI (interactive docs) | http://localhost:8000/docs |

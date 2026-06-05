# GKE Deployment

This folder contains Kubernetes manifests for deploying the RAG app to GKE at:

```text
https://chat.mammas.studio
```

## 1. Build and push the image

Everything here defaults to `europe-west3`. Replace `PROJECT_ID` with your Google Cloud project id.

```bash
gcloud artifacts repositories create debate-platform-repository \
  --repository-format=docker \
  --location=europe-west3 \
  --description="RAG LangChain images"

gcloud auth configure-docker europe-west3-docker.pkg.dev

docker build -t europe-west3-docker.pkg.dev/PROJECT_ID/debate-platform-repository/rag-langchain:latest .
docker push europe-west3-docker.pkg.dev/PROJECT_ID/debate-platform-repository/rag-langchain:latest
```

Then update the image in `deployment.yaml`:

```text
europe-west3-docker.pkg.dev/PROJECT_ID/debate-platform-repository/rag-langchain:latest
```

## Cloud Build trigger

The repository includes `cloudbuild.yaml` for the `rag-trigger` trigger. It builds the Docker image, pushes it to Artifact Registry in `europe-west3`, applies the Kubernetes manifests, and waits for the rollout.

Default substitutions:

```text
_REGION=europe-west3
_GKE_CLUSTER=rag-langchain
_GKE_NAMESPACE=rag-langchain
_AR_REPOSITORY=debate-platform-repository
_SERVICE_NAME=rag-langchain
```

If your GKE cluster name is different, update `_GKE_CLUSTER` in the trigger substitutions.

## 2. Create the runtime secret

Do not apply `secret.example.yaml` directly with the placeholder value. Create the real secret with:

```bash
kubectl create namespace rag-langchain

kubectl create secret generic rag-langchain-secrets \
  --namespace rag-langchain \
  --from-literal=OPENROUTER_API_KEY='your-openrouter-api-key'
```

## 3. Deploy to GKE

```bash
kubectl apply -k k8s
```

Check rollout status:

```bash
kubectl rollout status deployment/rag-langchain -n rag-langchain
kubectl get ingress rag-langchain -n rag-langchain
```

## 4. Point DNS to the GKE ingress

After the ingress receives an external IP, create an `A` record:

```text
chat.mammas.studio -> INGRESS_EXTERNAL_IP
```

The Google-managed certificate can take several minutes to become active after DNS is correct.

## Notes

- The app exposes `/healthz` for GKE health checks.
- The ingress uses a Google-managed TLS certificate for `chat.mammas.studio`.
- `BackendConfig.timeoutSec` is set to `300` so Server-Sent Events responses have enough time.
- The image currently includes the checked-in `RAG/vector_db/corpus.jsonl`. If the Qdrant index is not included or generated, dense retrieval is skipped and the app falls back to BM25 retrieval.

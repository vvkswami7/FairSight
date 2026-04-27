#!/bin/bash
# FairSight - Google Cloud Run + Firebase Deployment Script
set -e

PROJECT_ID="${GCP_PROJECT_ID:-project-760d5d90-96a4-4677-967}"
REGION="us-central1"
SERVICE_NAME="fairsight-api"
IMAGE="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Deploying FairSight backend to Google Cloud Run..."

# Build and push Docker image
echo "📦 Building Docker image..."
gcloud builds submit --tag $IMAGE .

# Deploy to Cloud Run
echo "☁️  Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 60 \
  --set-env-vars "GEMINI_API_KEY=${GEMINI_API_KEY}" \
  --project $PROJECT_ID

# Get the live service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --format 'value(status.url)' \
  --project $PROJECT_ID)

echo ""
echo "✅ Backend live at: $SERVICE_URL"
echo ""

# ── Fix 1: Auto-inject Cloud Run URL into frontend before deploy ──────────────
echo "🔧 Injecting Cloud Run URL into frontend/index.html..."
FRONTEND_FILE="../frontend/index.html"

if [ -f "$FRONTEND_FILE" ]; then
  # Replace the __CLOUD_RUN_URL__ placeholder with the real URL
  sed -i "s|window.__CLOUD_RUN_URL__ || ''|'$SERVICE_URL'|g" "$FRONTEND_FILE"
  echo "   Injected: $SERVICE_URL"
else
  echo "⚠️  frontend/index.html not found at $FRONTEND_FILE"
  echo "   Manually set the API URL to: $SERVICE_URL"
fi

# ── Deploy frontend to Firebase ───────────────────────────────────────────────
echo ""
echo "🌐 Deploying frontend to Firebase Hosting..."
cd ../frontend

if ! command -v firebase &> /dev/null; then
  echo "⚠️  Firebase CLI not found. Install with: npm install -g firebase-tools"
  echo "   Then run: firebase deploy --only hosting"
else
  firebase deploy --only hosting
fi

echo ""
echo "🎉 Full deployment complete!"
echo "   Backend API : $SERVICE_URL"
echo "   Health check: $SERVICE_URL/health"
echo "   Frontend    : Check Firebase Hosting dashboard"
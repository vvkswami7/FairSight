from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from routes import analysis, gemini_explain

app = FastAPI(title="FairSight API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://0.0.0.0:3000",
        "http://localhost:5173",  # Vite dev server
        "http://localhost:8000",  # Development
        "*"  # Allow public deployments (Cloud Run)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(gemini_explain.router, prefix="/api/explain", tags=["explain"])

@app.get("/health")
def health():
    return {"status": "ok", "service": "FairSight API"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)

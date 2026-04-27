from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from routes import analysis, gemini_explain

app = FastAPI(title="FairSight API", version="1.0.0")

# Mount static files - serve frontend from root URL
app.mount("/", StaticFiles(directory="static", html=True), name="static")

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

@app.get("/debug/gemini")
async def debug_gemini():
    import os
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return {"status": "error", "message": "API key not set"}
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say hello in one word"
        )
        return {"status": "ok", "response": response.text}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
def health():
    return {"status": "ok", "service": "FairSight API"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)

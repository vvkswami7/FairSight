from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.bias_engine import (
    detect_sensitive_columns, train_and_analyze, get_column_stats
)
from services.visualizer import generate_all_visualizations

router = APIRouter()

@router.post("/upload")
async def upload_and_analyze(
    file: UploadFile = File(...),
    label_col: str = Form(...),
    sensitive_cols: str = Form(...)  # comma-separated
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    df.columns = df.columns.str.strip()
    original_row_count = len(df)
    sampled = False
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42).reset_index(drop=True)
        sampled = True

    if df.empty or len(df) < 10:
        raise HTTPException(status_code=400, detail="Dataset must have at least 10 rows.")

    sensitive_list = [s.strip() for s in sensitive_cols.split(",") if s.strip()]

    # Validate columns
    for col in [label_col] + sensitive_list:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found in dataset.")

    try:
        results = train_and_analyze(df, label_col, sensitive_list)
        col_stats = get_column_stats(df)
        results["column_stats"] = col_stats
        results["columns"] = list(df.columns)
        results["filename"] = file.filename
        results["original_row_count"] = original_row_count
        results["sampled"] = sampled
        results["sampled_to"] = 10000 if sampled else original_row_count
        
        # Generate visualizations (base64-encoded charts)
        visualizations = generate_all_visualizations(results)
        results["visualizations"] = visualizations
        
        return JSONResponse(content=results)
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)} | Trace: {traceback.format_exc()[-500:]}"
        )


@router.post("/preview")
async def preview_dataset(file: UploadFile = File(...)):
    """Return columns and auto-detected sensitive columns for preview before analysis."""
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    sensitive = detect_sensitive_columns(df)

    return {
        "columns": list(df.columns),
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "detected_sensitive": sensitive,
        "sample": df.head(5).to_dict(orient="records"),
    }

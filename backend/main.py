import os
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile
import shutil
from dotenv import load_dotenv

from ocr_service.parser import DotsOCRParser

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Dots OCR API",
    description="API for document OCR parsing with layout detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize parser
parser = DotsOCRParser()


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Dots OCR API",
        "version": "1.0.0",
        "description": "Document OCR parsing with layout detection",
        "endpoints": {
            "parse_file": "/parse",
            "health": "/health",
            "config": "/config"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Dots OCR API"
    }


@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "vllm_ip": parser.ip,
        "vllm_port": parser.port,
        "model_name": parser.model_name,
        "temperature": parser.temperature,
        "top_p": parser.top_p,
        "max_completion_tokens": parser.max_completion_tokens,
        "num_thread": parser.num_thread,
        "dpi": parser.dpi,
        "output_dir": parser.output_dir,
        "min_pixels": parser.min_pixels,
        "max_pixels": parser.max_pixels,
    }


@app.post("/parse")
async def parse_file(
    file: UploadFile = File(...),
    prompt_mode: str = Form("prompt_layout_all_en"),
    bbox: str = Form(None),
):
    """
    Parse a PDF or image file for document layout and text extraction.
    
    Parameters:
    - file: The PDF or image file to parse
    - prompt_mode: The prompt mode to use (default: prompt_layout_all_en)
    - bbox: Optional bounding box as JSON string [x1, y1, x2, y2] for grounding OCR
    
    Returns:
    - JSON with parsing results including layout info, images, and markdown content
    """
    temp_dir = None
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Create temporary directory for input
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Parse bbox if provided
        parsed_bbox = None
        if bbox:
            try:
                parsed_bbox = json.loads(bbox)
                if not isinstance(parsed_bbox, list) or len(parsed_bbox) != 4:
                    raise ValueError("bbox must be a list of 4 numbers [x1, y1, x2, y2]")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid bbox format. Must be valid JSON")
        
        # Parse the file
        results = parser.parse_file(
            temp_file_path,
            prompt_mode=prompt_mode,
            bbox=parsed_bbox,
        )
        
        # Prepare response
        response_data = {
            "status": "success",
            "filename": file.filename,
            "prompt_mode": prompt_mode,
            "results": results,
        }
        
        return JSONResponse(content=response_data)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@app.post("/parse-batch")
async def parse_batch(
    files: list[UploadFile] = File(...),
    prompt_mode: str = Form("prompt_layout_all_en"),
):
    """
    Parse multiple PDF or image files in batch.
    
    Parameters:
    - files: List of PDF or image files to parse
    - prompt_mode: The prompt mode to use (default: prompt_layout_all_en)
    
    Returns:
    - JSON with parsing results for all files
    """
    temp_dir = None
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Create temporary directory for input
        temp_dir = tempfile.mkdtemp()
        
        batch_results = []
        
        for file in files:
            if not file.filename:
                continue
            
            temp_file_path = os.path.join(temp_dir, file.filename)
            
            # Save uploaded file
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            try:
                # Parse the file
                results = parser.parse_file(
                    temp_file_path,
                    prompt_mode=prompt_mode,
                )
                
                batch_results.append({
                    "filename": file.filename,
                    "status": "success",
                    "results": results,
                })
            except Exception as e:
                batch_results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e),
                })
        
        return JSONResponse(content={
            "status": "completed",
            "total_files": len(files),
            "results": batch_results,
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")
    
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import uvicorn
    import sys

    # Get host and port from environment or use defaults
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))

    # Check for command-line arguments
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv[1:], 1):
            if arg == "--port" and i < len(sys.argv) - 1:
                port = int(sys.argv[i + 1])
            elif arg == "--host" and i < len(sys.argv) - 1:
                host = sys.argv[i + 1]

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


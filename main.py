import os
import uuid
import shutil
import time
import json
import base64
import tempfile
import re
import logging
from pathlib import Path
from typing import Optional, List
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Third-party imports
from openai import OpenAI
from PIL import Image
from src.auth import verify_credentials, create_session, verify_session

# Import chart similarity analysis functions from your core module
try:
    from src.chart_similarity_cv import find_most_similar_charts_in_video, prepare_results_for_json
    COMPUTER_VISION_AVAILABLE = True
except ImportError:
    COMPUTER_VISION_AVAILABLE = False
    logging.warning("Computer vision module not available. GPT Vision only mode.")

# Create FastAPI app
app = FastAPI(title="Advanced Chart Analysis System")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY :
    client = OpenAI(api_key=OPENAI_API_KEY)
    GPT_VISION_AVAILABLE = True
else:
    client = None
    GPT_VISION_AVAILABLE = False
    logging.warning("OpenAI API key not configured. Computer vision only mode.")

# Create directories
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
STATIC_DIR = Path("static")
HISTORICAL_DIR = Path("historical")
TEMP_DIR = Path("temp")

for directory in [UPLOAD_DIR, RESULTS_DIR, STATIC_DIR, HISTORICAL_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)

# Create category directories for computer vision
for category in ["gold", "btc", "usdcad"]:
    (RESULTS_DIR / category).mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results", StaticFiles(directory="results"), name="results")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/historical", StaticFiles(directory="historical"), name="historical")
app.mount("/temp", StaticFiles(directory="temp"), name="temp")

# Store job status in memory (for both analysis types)
jobs = {}

# Pydantic models
class AnalysisParams(BaseModel):
    fps: float = 1.0
    detect_color: str = "green"
    category: Optional[str] = None
    analysis_type: str = "computer_vision"  # "computer_vision" or "gpt_vision"

class ComparisonResult(BaseModel):
    year: str
    score: int
    analysis: str
    historical_image_url: str = ""

class ComparisonResponse(BaseModel):
    results: List[ComparisonResult]
    top_matches: List[ComparisonResult]
    current_image_url: str = ""

# GPT Vision helper functions
def generate_prompt(historical_year: str) -> str:
    return f"""
You are a financial chart analyst.

Please compare two candlestick charts of **Bitcoin**. 

The first chart shows Bitcoin's candlestick data from **2025 YTD**.
The second chart shows Bitcoin's candlestick data from the year **{historical_year}**.

Compare them based on:
1. **Price Trends** – uptrends, downtrends, or sideways movement.
2. **Volatility** – based on candlestick size and frequency.
3. **Patterns** – consolidation zones, breakouts, or reversals.
4. **Shape Similarity** – general structure and trend resemblance.

At the end, assign a **similarity score from 0 to 100** and explain why.
"""

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def validate_image(file_path: str) -> bool:
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except:
        return False

def format_analysis_text(analysis_text: str) -> str:
    """Format the analysis text for better readability"""
    text = analysis_text.strip()
    
    # Replace ### with proper headers
    text = re.sub(r'### (.*?)\n', r'<h4>\1</h4>\n', text)
    text = re.sub(r'##+ (.*?)\n', r'<h4>\1</h4>\n', text)
    
    # Format numbered points
    text = re.sub(r'\n(\d+\.)\s*\*\*(.*?)\*\*:?\s*-?\s*', r'\n<p><strong>\1 \2:</strong> ', text)
    text = re.sub(r'\n(\d+\.)\s*(.*?):\s*', r'\n<p><strong>\1 \2:</strong> ', text)
    
    # Format bullet points
    text = re.sub(r'\n-\s*\*\*(.*?)\*\*:?\s*', r'\n<p><strong>• \1:</strong> ', text)
    text = re.sub(r'\n-\s*(.*?):\s*', r'\n<p><strong>• \1:</strong> ', text)
    
    # Bold important terms
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    
    # Highlight similarity score
    text = re.sub(r'(\*\*Score:?\s*(\d+)/100\*\*)', r'<span class="score-highlight">Score: \2/100</span>', text)
    text = re.sub(r'(Score:?\s*(\d+)/100)', r'<span class="score-highlight">Score: \2/100</span>', text)
    
    # Format paragraphs
    paragraphs = text.split('\n\n')
    formatted_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if para:
            if not para.startswith('<h4>') and not para.startswith('<p>') and not para.startswith('<span'):
                para = f'<p>{para}</p>'
            formatted_paragraphs.append(para)
    
    return '\n'.join(formatted_paragraphs)

def update_cv_progress(job_id, progress):
    """Update computer vision job progress"""
    if job_id in jobs:
        jobs[job_id]["progress"] = progress

# API Routes

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Return the main HTML file"""
    return FileResponse("static/index.html")

@app.get("/admin", response_class=HTMLResponse)
async def get_admin():
    """Return the admin panel HTML file"""
    return FileResponse("static/admin.html")

@app.get("/api/system-status")
async def get_system_status():
    """Get available analysis methods"""
    return {
        "computer_vision_available": COMPUTER_VISION_AVAILABLE,
        "gpt_vision_available": GPT_VISION_AVAILABLE,
        "historical_charts_count": len([f for f in os.listdir(HISTORICAL_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
    }

# Computer Vision Routes
@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    category: str = Form(None),
    analysis_type: str = Form("computer_vision")
):
    """Handle video upload for computer vision analysis"""
    if analysis_type == "computer_vision" and not COMPUTER_VISION_AVAILABLE:
        raise HTTPException(status_code=400, detail="Computer vision analysis not available")
    
    # Validate video file for computer vision
    if analysis_type == "computer_vision":
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a video file.")
        
        # Validate category
        if category not in ["gold", "btc", "usdcad", None]:
            raise HTTPException(status_code=400, detail="Invalid category. Must be 'gold', 'btc', or 'usdcad'.")
    
    # Generate unique ID for this job
    job_id = str(uuid.uuid4())
    
    # Create job directory in uploads
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    
    # Save uploaded file
    file_path = job_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create job entry
    jobs[job_id] = {
        "id": job_id,
        "status": "uploaded",
        "filename": file.filename,
        "file_path": str(file_path),
        "created_at": time.time(),
        "results": None,
        "progress": 0,
        "category": category,
        "analysis_type": analysis_type
    }
    
    return {
        "job_id": job_id, 
        "filename": file.filename, 
        "status": "uploaded", 
        "category": category,
        "analysis_type": analysis_type
    }

@app.post("/api/analyze/{job_id}")
async def analyze_video(job_id: str, background_tasks: BackgroundTasks, params: AnalysisParams):
    """Start video analysis in background (computer vision)"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if params.analysis_type == "computer_vision" and not COMPUTER_VISION_AVAILABLE:
        raise HTTPException(status_code=400, detail="Computer vision analysis not available")
    
    # Update job status
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["params"] = params.dict()
    jobs[job_id]["progress"] = 0
    jobs[job_id]["analysis_type"] = params.analysis_type
    
    # If category was provided in params, update job category
    if params.category:
        jobs[job_id]["category"] = params.category
    
    # Create results directory
    result_dir = RESULTS_DIR / job_id
    result_dir.mkdir(exist_ok=True)
    
    # Run analysis in background
    background_tasks.add_task(
        run_computer_vision_analysis, 
        job_id=job_id,
        file_path=jobs[job_id]["file_path"],
        output_dir=str(result_dir),
        fps=params.fps,
        category=jobs[job_id].get("category")
    )
    
    return {
        "job_id": job_id, 
        "status": "processing", 
        "category": jobs[job_id].get("category"),
        "analysis_type": params.analysis_type
    }

def run_computer_vision_analysis(job_id: str, file_path: str, output_dir: str, fps: float, category: str = None):
    """Run computer vision analysis in background"""
    try:
        # Progress callback function
        def progress_callback(progress):
            update_cv_progress(job_id, progress)
        
        # Run analysis
        results = find_most_similar_charts_in_video(
            video_path=file_path,
            output_dir=output_dir,
            fps=fps,
            progress_callback=progress_callback
        )
        
        # Prepare results for JSON
        serializable_results = prepare_results_for_json(results)
        
        # Add category to results
        if category:
            serializable_results["category"] = category
            serializable_results["analysis_type"] = "computer_vision"
        
        # Save results to file
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(serializable_results, f)
        
        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["results"] = serializable_results
        jobs[job_id]["progress"] = 100
        
    except Exception as e:
        # Update job status with error
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        logger.error(f"Error processing computer vision job {job_id}: {str(e)}")

# NEW GPT Vision Routes (Job-based with HTTP Polling)
@app.post("/api/gpt-upload")
async def gpt_upload_image(file: UploadFile = File(...)):
    """Upload image for GPT Vision analysis (job-based)"""
    if not GPT_VISION_AVAILABLE:
        raise HTTPException(status_code=400, detail="GPT Vision analysis not available")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Create job directory
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    
    # Save uploaded file
    file_path = job_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create job entry
    jobs[job_id] = {
        "id": job_id,
        "status": "uploaded",
        "filename": file.filename,
        "file_path": str(file_path),
        "created_at": time.time(),
        "results": None,
        "progress": 0,
        "analysis_type": "gpt_vision"
    }
    
    return {
        "job_id": job_id,
        "filename": file.filename,
        "status": "uploaded",
        "analysis_type": "gpt_vision"
    }

@app.post("/api/gpt-analyze/{job_id}")
async def gpt_analyze_image(job_id: str, background_tasks: BackgroundTasks):
    """Start GPT Vision analysis in background"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if not GPT_VISION_AVAILABLE:
        raise HTTPException(status_code=400, detail="GPT Vision analysis not available")
    
    # Update job status
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["progress"] = 0
    
    # Run analysis in background
    background_tasks.add_task(run_gpt_vision_analysis, job_id)
    
    return {
        "job_id": job_id,
        "status": "processing",
        "analysis_type": "gpt_vision"
    }

def run_gpt_vision_analysis(job_id: str):
    """Run GPT Vision analysis in background"""
    try:
        file_path = jobs[job_id]["file_path"]
        
        logger.info(f"Starting GPT comparison for job: {job_id}")
        
        # Update progress
        jobs[job_id]["progress"] = 5
        
        if not validate_image(file_path):
            raise Exception("Invalid image file")
        
        jobs[job_id]["progress"] = 10
        query_b64 = encode_image_base64(file_path)
        
        jobs[job_id]["progress"] = 15
        
        if not os.path.exists(HISTORICAL_DIR):
            raise Exception("Historical data folder not found")
        
        historical_files = [f for f in os.listdir(HISTORICAL_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if not historical_files:
            raise Exception("No historical data found")
        
        total_files = len(historical_files)
        logger.info(f"Found {total_files} historical files")
        
        jobs[job_id]["progress"] = 20
        
        results = []
        
        for index, file_name in enumerate(historical_files):
            year = os.path.splitext(file_name)[0]
            historical_path = os.path.join(HISTORICAL_DIR, file_name)
            
            # Update progress (20% to 90%)
            progress = 20 + int((index / total_files) * 70)
            jobs[job_id]["progress"] = progress
            
            try:
                historical_b64 = encode_image_base64(historical_path)
                prompt = generate_prompt(year)
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a financial chart analyst."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{query_b64}"}},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{historical_b64}"}}
                            ]
                        }
                    ],
                    max_tokens=1000
                )
                
                reply = response.choices[0].message.content
                match = re.search(r'(\d{1,3})\s*/?\s*100', reply)
                score = int(match.group(1)) if match else 0
                
                results.append({
                    "year": year,
                    "score": score,
                    "analysis": format_analysis_text(reply),
                    "historical_image_url": f"/historical/{file_name}"
                })
                
                logger.info(f"Completed {year}: {score}/100")
                
            except Exception as e:
                logger.error(f"Error with {year}: {str(e)}")
                continue
        
        jobs[job_id]["progress"] = 92
        
        # Sort results
        results = sorted([r for r in results if r["score"] is not None], 
                        key=lambda x: x["score"], reverse=True)
        
        top_matches = results[:5]
        
        # Copy uploaded file to temp for frontend access
        temp_filename = f"{job_id}.png"
        temp_file_path = os.path.join(TEMP_DIR, temp_filename)
        shutil.copy2(file_path, temp_file_path)
        
        final_results = {
            "results": results,
            "top_matches": top_matches,
            "current_image_url": f"/temp/{temp_filename}",
            "analysis_type": "gpt_vision"
        }
        
        # Update job with results
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["results"] = final_results
        jobs[job_id]["progress"] = 100
        
        logger.info(f"GPT analysis completed! Best match: {top_matches[0]['year']} ({top_matches[0]['score']}/100)")
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        logger.error(f"Error processing GPT job {job_id}: {str(e)}")

# OLD GPT Vision Route (keep for backward compatibility)
@app.post("/api/gpt-compare", response_model=ComparisonResponse)
async def gpt_compare_charts(file: UploadFile = File(...)):
    """GPT Vision chart comparison (old WebSocket version - kept for compatibility)"""
    if not GPT_VISION_AVAILABLE:
        raise HTTPException(status_code=400, detail="GPT Vision analysis not available")
    
    logger.info(f"Starting GPT comparison for: {file.filename}")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file to temp folder with unique name
    file_id = str(uuid.uuid4())
    temp_filename = f"{file_id}.png"
    temp_file_path = os.path.join(TEMP_DIR, temp_filename)
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        if not validate_image(temp_file_path):
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        query_b64 = encode_image_base64(temp_file_path)
        
        if not os.path.exists(HISTORICAL_DIR):
            raise HTTPException(status_code=404, detail="Historical data folder not found")
        
        historical_files = [f for f in os.listdir(HISTORICAL_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if not historical_files:
            raise HTTPException(status_code=404, detail="No historical data found")
        
        total_files = len(historical_files)
        logger.info(f"Found {total_files} historical files")
        
        results = []
        
        for index, file_name in enumerate(historical_files):
            year = os.path.splitext(file_name)[0]
            historical_path = os.path.join(HISTORICAL_DIR, file_name)
            
            try:
                historical_b64 = encode_image_base64(historical_path)
                prompt = generate_prompt(year)
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a financial chart analyst."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{query_b64}"}},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{historical_b64}"}}
                            ]
                        }
                    ],
                    max_tokens=1000
                )
                
                reply = response.choices[0].message.content
                match = re.search(r'(\d{1,3})\s*/?\s*100', reply)
                score = int(match.group(1)) if match else 0
                
                results.append({
                    "year": year,
                    "score": score,
                    "analysis": format_analysis_text(reply),
                    "historical_image_url": f"/historical/{file_name}"
                })
                
                logger.info(f"Completed {year}: {score}/100")
                
            except Exception as e:
                logger.error(f"Error with {year}: {str(e)}")
                continue
        
        results = sorted([r for r in results if r["score"] is not None], 
                        key=lambda x: x["score"], reverse=True)
        
        top_matches = results[:5]
        
        logger.info(f"GPT analysis completed! Best match: {top_matches[0]['year']} ({top_matches[0]['score']}/100)")
        
        return ComparisonResponse(
            results=results, 
            top_matches=top_matches,
            current_image_url=f"/temp/{temp_filename}"
        )
        
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise

# Common Routes
@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and results"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    response = {
        "id": job["id"],
        "status": job["status"],
        "filename": job["filename"],
        "progress": job.get("progress", 0),
        "category": job.get("category"),
        "analysis_type": job.get("analysis_type", "computer_vision")
    }
    
    # Add results if job is completed
    if job["status"] == "completed" and job.get("results"):
        # Add category and analysis type to results if available
        if "category" in job and job["category"]:
            if isinstance(job["results"], dict):
                job["results"]["category"] = job["category"]
                job["results"]["analysis_type"] = job.get("analysis_type", "computer_vision")
        
        response["results"] = job["results"]
    
    # Add error if job failed
    if job["status"] == "failed" and "error" in job:
        response["error"] = job["error"]
    
    return response


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs"""
    job_list = []
    for job_id, job in jobs.items():
        job_list.append({
            "id": job["id"],
            "status": job["status"],
            "filename": job["filename"],
            "created_at": job["created_at"],
            "category": job.get("category"),
            "analysis_type": job.get("analysis_type", "computer_vision")
        })
    
    # Sort by created_at (newest first)
    job_list.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {"jobs": job_list}

@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Remove from memory
    job = jobs.pop(job_id)
    
    # Remove files
    job_upload_dir = UPLOAD_DIR / job_id
    job_result_dir = RESULTS_DIR / job_id
    
    if job_upload_dir.exists():
        shutil.rmtree(job_upload_dir)
    
    if job_result_dir.exists():
        shutil.rmtree(job_result_dir)
    
    return {"status": "deleted", "job_id": job_id}


@app.post("/api/admin/login")
async def admin_login(username: str = Form(...), password: str = Form(...)):
    if verify_credentials(username, password):
        token = create_session(username)
        return {"success": True, "token": token}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/api/admin/verify")
async def verify_admin_session(token: str = Form(...)):
    username = verify_session(token)
    if username:
        return {"success": True, "username": username}
    else:
        raise HTTPException(status_code=401, detail="Invalid session")
    
# Admin Routes
@app.get("/api/historical")
async def list_historical_data():
    """List historical chart data for GPT Vision"""
    if not os.path.exists(HISTORICAL_DIR):
        return {"files": []}
    
    files = [f for f in os.listdir(HISTORICAL_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    files.sort()  # Sort alphabetically
    return {"files": files}

@app.post("/api/upload-historical")
async def upload_historical_data(year: str = Form(...), file: UploadFile = File(...)):
    """Upload historical chart data for GPT Vision"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate year
    try:
        year_int = int(year)
        if year_int < 1900 or year_int > 2030:
            raise ValueError("Year out of range")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid year format")
    
    if not os.path.exists(HISTORICAL_DIR):
        os.makedirs(HISTORICAL_DIR)
    
    # Use .png extension for consistency
    file_path = os.path.join(HISTORICAL_DIR, f"{year}.png")
    
    # Convert and save as PNG
    try:
        # Open and convert image
        with Image.open(file.file) as img:
            # Convert to RGB if necessary
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            # Save as PNG
            img.save(file_path, "PNG", optimize=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
    
    return {"message": f"Historical data for year {year} uploaded successfully", "filename": f"{year}.png"}

@app.delete("/api/historical/{filename}")
async def delete_historical_data(filename: str):
    """Delete historical chart data"""
    file_path = os.path.join(HISTORICAL_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Validate filename to prevent path traversal
    if not filename.endswith(('.png', '.jpg', '.jpeg')) or '/' in filename or '\\' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    try:
        os.unlink(file_path)
        return {"message": f"File {filename} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

@app.get("/api/cleanup")
async def cleanup_old_jobs():
    """Cleanup old jobs (older than 24 hours)"""
    current_time = time.time()
    removed_count = 0
    
    # Find jobs to remove
    jobs_to_remove = []
    for job_id, job in jobs.items():
        if current_time - job["created_at"] > 24 * 60 * 60:  # 24 hours
            jobs_to_remove.append(job_id)
    
    # Remove jobs
    for job_id in jobs_to_remove:
        # Remove from memory
        if job_id in jobs:
            del jobs[job_id]
        
        # Remove files
        job_upload_dir = UPLOAD_DIR / job_id
        job_result_dir = RESULTS_DIR / job_id
        
        if job_upload_dir.exists():
            shutil.rmtree(job_upload_dir)
        
        if job_result_dir.exists():
            shutil.rmtree(job_result_dir)
            
        removed_count += 1
    
    return {"removed": removed_count}

@app.delete("/api/clear-all-jobs")
async def clear_all_jobs():
    """Clear all jobs (admin function)"""
    removed_count = len(jobs)
    
    # Get all job IDs
    job_ids = list(jobs.keys())
    
    # Clear jobs from memory
    jobs.clear()
    
    # Remove all job files
    for job_id in job_ids:
        job_upload_dir = UPLOAD_DIR / job_id
        job_result_dir = RESULTS_DIR / job_id
        
        if job_upload_dir.exists():
            shutil.rmtree(job_upload_dir)
        
        if job_result_dir.exists():
            shutil.rmtree(job_result_dir)
    
    return {"removed": removed_count, "message": f"Cleared {removed_count} jobs"}

@app.get("/api/admin/stats")
async def get_admin_stats():
    """Get admin statistics"""
    # Count jobs by status
    total_jobs = len(jobs)
    active_jobs = sum(1 for job in jobs.values() if job["status"] == "processing")
    completed_jobs = sum(1 for job in jobs.values() if job["status"] == "completed")
    failed_jobs = sum(1 for job in jobs.values() if job["status"] == "failed")
    
    # Count historical charts
    historical_count = 0
    if os.path.exists(HISTORICAL_DIR):
        historical_count = len([f for f in os.listdir(HISTORICAL_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Calculate disk usage
    def get_dir_size(path):
        total = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total += os.path.getsize(fp)
        except:
            pass
        return total
    
    upload_size = get_dir_size(UPLOAD_DIR)
    results_size = get_dir_size(RESULTS_DIR)
    historical_size = get_dir_size(HISTORICAL_DIR)
    temp_size = get_dir_size(TEMP_DIR)
    
    return {
        "jobs": {
            "total": total_jobs,
            "active": active_jobs,
            "completed": completed_jobs,
            "failed": failed_jobs
        },
        "historical_charts": historical_count,
        "disk_usage": {
            "uploads_mb": round(upload_size / (1024 * 1024), 2),
            "results_mb": round(results_size / (1024 * 1024), 2),
            "historical_mb": round(historical_size / (1024 * 1024), 2),
            "temp_mb": round(temp_size / (1024 * 1024), 2),
            "total_mb": round((upload_size + results_size + historical_size + temp_size) / (1024 * 1024), 2)
        },
        "system_status": {
            "computer_vision_available": COMPUTER_VISION_AVAILABLE,
            "gpt_vision_available": GPT_VISION_AVAILABLE
        }
    }

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting Advanced Chart Analysis System...")
    logger.info(f"Computer Vision Available: {COMPUTER_VISION_AVAILABLE}")
    logger.info(f"GPT Vision Available: {GPT_VISION_AVAILABLE}")
    
    # Create admin.html file if it doesn't exist
    admin_html_path = STATIC_DIR / "admin.html"
    if not admin_html_path.exists():
        logger.info("Creating admin.html file...")
        # You'll need to save the admin panel HTML content to this file
        # For now, we'll create a placeholder
        with open(admin_html_path, "w") as f:
            f.write("<!-- Admin panel HTML content goes here -->")
    
    if os.path.exists(HISTORICAL_DIR):
        files = [f for f in os.listdir(HISTORICAL_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
        logger.info(f"Found {len(files)} historical chart files for GPT Vision")
    
    # Clean up old temp files on startup
    if os.path.exists(TEMP_DIR):
        try:
            for file in os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, file)
                if os.path.isfile(file_path):
                    # Remove files older than 1 hour
                    if time.time() - os.path.getmtime(file_path) > 3600:
                        os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Error cleaning temp files: {e}")
    
    logger.info("Application ready")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "computer_vision_available": COMPUTER_VISION_AVAILABLE,
        "gpt_vision_available": GPT_VISION_AVAILABLE,
        "active_jobs": len([job for job in jobs.values() if job["status"] == "processing"]),
        "total_jobs": len(jobs)
    }

if __name__ == "__main__":
    print("Starting Advanced Chart Analysis System...")
    print(f"Computer Vision Available: {COMPUTER_VISION_AVAILABLE}")
    print(f"GPT Vision Available: {GPT_VISION_AVAILABLE}")
    print("Main app: http://localhost:5000")
    print("Admin panel: http://localhost:5000/admin")
    uvicorn.run(app, host="0.0.0.0", port=5000)

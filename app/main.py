from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
from typing import List, Dict, Optional
# Import the modules - try both app-relative and local imports
try:
    # When running from project root
    from app.resume_processing import extract_text_from_pdfs
    from app.llm_shortlister import shortlist_resumes, EVAL_WEIGHTS
except ImportError:
    # When running from app directory
    from resume_processing import extract_text_from_pdfs
    from llm_shortlister import shortlist_resumes, EVAL_WEIGHTS
import markdown
import asyncio
from io import BytesIO

app = FastAPI()

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Landing page: choose local folder or provide Google Drive link."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-local", response_class=HTMLResponse)
async def upload_local(request: Request, files: List[UploadFile] = File(...)):
    """Handle multiple PDF uploads from local folder."""
    saved_files = []
    for file in files:
        if file.filename.lower().endswith('.pdf'):
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved_files.append(file.filename)
    if not saved_files:
        return templates.TemplateResponse("index.html", {"request": request, "error": "No valid PDF files uploaded."})
    # Placeholder: Show uploaded files (next: process them)
    return templates.TemplateResponse("upload_success.html", {"request": request, "files": saved_files})

@app.post("/upload-drive")
def upload_drive():
    # To be implemented: handle Google Drive link
    return {"status": "Not implemented"}

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_resumes(request: Request):
    """Extract text from uploaded PDFs and show a summary (next: LLM analysis)."""
    pdf_dir = UPLOAD_DIR
    # Update to capture both return values
    pdf_texts, resume_data_dict = extract_text_from_pdfs(pdf_dir)
    # For now, just show the filenames and a text snippet for each
    summary = [
        {
            "filename": fname,
            "snippet": (text[:500] + "...") if len(text) > 500 else text
        }
        for fname, text in pdf_texts.items()
    ]
    return templates.TemplateResponse("analyze_summary.html", {"request": request, "summary": summary})

@app.post("/shortlist", response_class=HTMLResponse)
async def shortlist_candidates(
    request: Request,
    job_description: Optional[str] = Form(None),
    weights_technical_expertise: Optional[int] = Form(35),
    weights_project_complexity: Optional[int] = Form(15),
    weights_hackathon_experience: Optional[int] = Form(25),
    weights_work_ethic: Optional[int] = Form(10),
    weights_extracurricular: Optional[int] = Form(10),
    weights_certifications: Optional[int] = Form(5),
    custom_weights_json: Optional[str] = Form(None)
):
    """Process uploaded resumes with optional job description and custom weights."""
    pdf_dir = UPLOAD_DIR
    
    # Extract text from PDFs
    pdf_texts, resume_data_dict = extract_text_from_pdfs(pdf_dir)
    
    if not pdf_texts:
        return templates.TemplateResponse("analyze_summary.html", {
            "request": request, 
            "summary": [], 
            "error": "No resumes found for analysis."
        })
    
    # Process standard weights
    standard_weights = {
        "technical_expertise": weights_technical_expertise / 100,
        "project_complexity": weights_project_complexity / 100,
        "hackathon_experience": weights_hackathon_experience / 100,
        "work_ethic": weights_work_ethic / 100,
        "extracurricular": weights_extracurricular / 100,
        "certifications": weights_certifications / 100
    }
    
    # Process custom weights if provided
    all_weights = dict(standard_weights)
    if custom_weights_json and custom_weights_json.strip():
        try:
            # Parse JSON string into a list of dictionaries
            custom_weights = json.loads(custom_weights_json)
            
            # Process each custom weight
            for weight in custom_weights:
                # Convert name to snake_case format to use as a key
                key = weight['name'].lower().replace(' ', '_')
                value = weight['value']
                
                # Add to weights dictionary
                all_weights[key] = value
                
            # Normalize weights to ensure they sum to 1.0
            total = sum(all_weights.values())
            if abs(total - 1.0) > 0.01:  # Allow small rounding errors
                for key in all_weights:
                    all_weights[key] = all_weights[key] / total
                    
        except Exception as e:
            print(f"Error processing custom weights: {str(e)}")
            # Fall back to standard weights
            all_weights = dict(standard_weights)
    
    # Run LLM shortlisting with custom parameters
    shortlist_md = await shortlist_resumes(
        pdf_texts, 
        resume_data_dict, 
        eval_weights=all_weights,
        job_description=job_description
    )
    
    # Render markdown to HTML
    shortlist_html = markdown.markdown(shortlist_md, extensions=["extra"])
    
    return templates.TemplateResponse("shortlist_result.html", {
        "request": request,
        "shortlist_md": shortlist_md,
        "shortlist_html": shortlist_html
    })

@app.post("/download-shortlist")
async def download_shortlist(shortlist_md: str = Form(...)):
    md_bytes = shortlist_md.encode("utf-8")
    return StreamingResponse(BytesIO(md_bytes),
                             media_type="text/markdown",
                             headers={"Content-Disposition": "attachment; filename=shortlisted_candidates.md"})
import os
import pdfplumber
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import re
import time
import pypdfium2 as pdfium  # Backup extraction if pdfplumber fails
import logging
import json
from dataclasses import dataclass, asdict, field

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("resume_extraction.log"),
        logging.StreamHandler()
    ]
)

@dataclass
class ResumeData:
    """Structured data extracted from a resume."""
    filename: str
    full_text: str
    name: str = ""
    email: str = ""
    phone: str = ""
    github: str = ""
    linkedin: str = ""
    education: List[str] = field(default_factory=list)
    experience: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    projects: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

def clean_text(text: str) -> str:
    """Clean extracted text to remove common PDF extraction artifacts."""
    if not text:
        return ""
    
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove common PDF artifacts
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII chars
    text = re.sub(r'\s{2,}', ' ', text)         # Replace multiple spaces with single space
    
    return text.strip()

def extract_with_pdfium(file_path: str) -> str:
    """Backup extraction method using pdfium when pdfplumber fails."""
    try:
        pdf = pdfium.PdfDocument(file_path)
        text_parts = []
        
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            text_parts.append(textpage.get_text_range())
            
        return "\n\n".join(text_parts)
    except Exception as e:
        return f"[ERROR extracting with pdfium: {e}]"

def extract_contact_info(text: str) -> Dict[str, str]:
    """Extract contact information from resume text."""
    contact_info = {
        "name": "",
        "email": "",
        "phone": "",
        "github": "",
        "linkedin": ""
    }
    
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_matches = re.findall(email_pattern, text)
    if email_matches:
        contact_info["email"] = email_matches[0]
    
    # Phone pattern (handles various formats)
    phone_patterns = [
        r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',  # (123) 456-7890, 123-456-7890
        r'\b\d{10}\b',  # 1234567890
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'  # 123-456-7890
    ]
    
    for pattern in phone_patterns:
        phone_matches = re.findall(pattern, text)
        if phone_matches:
            contact_info["phone"] = phone_matches[0]
            break
    
    # GitHub profile
    github_patterns = [
        r'github\.com/([A-Za-z0-9_-]+)',
        r'github: ([A-Za-z0-9_-]+)',
        r'GitHub: ([A-Za-z0-9_-]+)',
        r'GitHub:?\s+(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_-]+)'
    ]
    
    for pattern in github_patterns:
        github_matches = re.findall(pattern, text)
        if github_matches:
            github_username = github_matches[0]
            contact_info["github"] = f"https://github.com/{github_username}"
            break
    
    # LinkedIn profile
    linkedin_patterns = [
        r'linkedin\.com/in/([A-Za-z0-9_-]+)',
        r'linkedin: ([A-Za-z0-9_-]+)',
        r'LinkedIn: ([A-Za-z0-9_-]+)',
        r'LinkedIn:?\s+(?:https?://)?(?:www\.)?linkedin\.com/in/([A-Za-z0-9_-]+)'
    ]
    
    for pattern in linkedin_patterns:
        linkedin_matches = re.findall(pattern, text)
        if linkedin_matches:
            linkedin_username = linkedin_matches[0]
            contact_info["linkedin"] = f"https://linkedin.com/in/{linkedin_username}"
            break
    
    # Name extraction - attempt to find name at the beginning of the resume
    # This is a heuristic approach that looks for capitalized words at the beginning
    first_line = text.split('\n')[0].strip()
    if first_line and len(first_line) < 50:  # Reasonable name length
        words = first_line.split()
        if 1 < len(words) < 5:  # Typical name has 2-4 words
            all_caps = all(w[0].isupper() for w in words if w)
            if all_caps:
                contact_info["name"] = first_line
    
    # If name not found in first line, look for lines that might contain only a name
    if not contact_info["name"]:
        # Look at first 10 lines for potential name
        for line in text.split('\n')[:10]:
            line = line.strip()
            if line and len(line) < 50:
                words = line.split()
                if 1 < len(words) < 5:
                    all_caps = all(w[0].isupper() for w in words if w)
                    no_special_chars = all(w.replace('-', '').replace("'", '').isalpha() for w in words if w)
                    if all_caps and no_special_chars and not any(w.lower() in ["resume", "cv", "curriculum", "vitae"] for w in words):
                        contact_info["name"] = line
                        break
    
    return contact_info

def extract_sections(text: str) -> Dict[str, List[str]]:
    """Extract key sections from resume text."""
    sections = {
        "education": [],
        "experience": [],
        "skills": [],
        "projects": [],
        "certifications": []
    }
    
    # Common section headers in resumes
    section_patterns = {
        "education": [r'EDUCATION', r'Education', r'ACADEMIC BACKGROUND', r'Academic Background'],
        "experience": [r'EXPERIENCE', r'Experience', r'WORK EXPERIENCE', r'Work Experience', r'EMPLOYMENT', r'Employment'],
        "skills": [r'SKILLS', r'Skills', r'TECHNICAL SKILLS', r'Technical Skills', r'TECHNOLOGIES', r'Technologies'],
        "projects": [r'PROJECTS', r'Projects', r'PERSONAL PROJECTS', r'Personal Projects'],
        "certifications": [r'CERTIFICATIONS', r'Certifications', r'CERTIFICATES', r'Certificates']
    }
    
    # Split text into lines for processing
    lines = text.split('\n')
    
    # Find section boundaries
    section_boundaries = []
    for i, line in enumerate(lines):
        for section, patterns in section_patterns.items():
            for pattern in patterns:
                if re.match(f'^{pattern}', line.strip()):
                    section_boundaries.append((i, section))
                    break
    
    # Sort boundaries by line number
    section_boundaries.sort()
    
    # Extract content between section boundaries
    for i, (start_line, section) in enumerate(section_boundaries):
        end_line = section_boundaries[i+1][0] if i+1 < len(section_boundaries) else len(lines)
        
        # Get content (skip header line)
        content = '\n'.join(lines[start_line+1:end_line]).strip()
        
        # Store in appropriate section
        if content:
            sections[section].append(content)
    
    return sections

def extract_single_pdf(file_path: str, filename: str) -> tuple:
    """
    Extract text and structured data from a single PDF file with fallback methods.
    Returns: (filename, extracted_text, resume_data)
    """
    try:
        start_time = time.time()
        
        # First try with pdfplumber
        with pdfplumber.open(file_path) as pdf:
            # Get total pages for progress calculation
            total_pages = len(pdf.pages)
            
            # Extract text from each page with progress tracking if many pages
            texts = []
            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        texts.append(page_text)
                except Exception as page_e:
                    logging.warning(f"Error extracting page {i+1}/{total_pages} from {filename}: {page_e}")
                    texts.append(f"[Error on page {i+1}]")
            
            text = "\n".join(texts)
            
        # If extraction yielded little or no text, try fallback method
        if not text or len(text) < 100:
            logging.info(f"Minimal text extracted from {filename} with pdfplumber, trying pdfium")
            text = extract_with_pdfium(file_path)
        
        cleaned_text = clean_text(text)
        
        # Extract structured information
        contact_info = extract_contact_info(cleaned_text)
        sections = extract_sections(cleaned_text)
        
        # Create structured resume data
        resume_data = ResumeData(
            filename=filename,
            full_text=cleaned_text,
            name=contact_info["name"],
            email=contact_info["email"],
            phone=contact_info["phone"],
            github=contact_info["github"],
            linkedin=contact_info["linkedin"],
            education=sections["education"],
            experience=sections["experience"],
            skills=sections["skills"],
            projects=sections["projects"],
            certifications=sections["certifications"]
        )
        
        extraction_time = time.time() - start_time
        
        # Log success or partial extraction issues
        if len(cleaned_text) < 100:
            logging.warning(f"Possible extraction issues with {filename}: only {len(cleaned_text)} chars extracted in {extraction_time:.2f}s")
        else:
            logging.debug(f"Successfully extracted {len(cleaned_text)} chars from {filename} in {extraction_time:.2f}s")
            
        return filename, cleaned_text, resume_data
        
    except Exception as e:
        logging.error(f"Failed to extract text from {filename}: {e}")
        return filename, f"[ERROR extracting text: {e}]", ResumeData(filename=filename, full_text=f"[ERROR: {e}]")

def extract_text_from_pdfs(pdf_dir: str, max_workers: Optional[int] = None) -> Tuple[Dict[str, str], Dict[str, ResumeData]]:
    """
    Extract text and structured data from all PDF files in the given directory using parallel processing.
    Returns a tuple of:
        - Dict mapping filenames to extracted text
        - Dict mapping filenames to structured ResumeData objects
    
    Args:
        pdf_dir: Directory containing PDF files
        max_workers: Maximum number of parallel workers (None = auto)
    """
    # Check if directory exists
    if not os.path.exists(pdf_dir):
        logging.error(f"Directory not found: {pdf_dir}")
        return {}, {}
        
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    total_files = len(pdf_files)
    
    if total_files == 0:
        logging.warning(f"No PDF files found in {pdf_dir}")
        return {}, {}
    
    logging.info(f"Starting extraction of {total_files} PDF files from {pdf_dir}")
    start_time = time.time()
    
    # Limit max_workers to avoid overwhelming system resources
    if max_workers is None:
        # Use the minimum of processor count or 8 to avoid overloading
        max_workers = min(os.cpu_count() or 4, 8)
    logging.info(f"Using {max_workers} parallel workers for extraction")
    
    # Use parallel processing for many PDFs
    pdf_texts = {}
    resume_data_dict = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create extraction tasks
        future_to_file = {
            executor.submit(
                extract_single_pdf, 
                os.path.join(pdf_dir, filename), 
                filename
            ): filename for filename in pdf_files
        }
        
        # Process results as they complete with progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                           total=len(pdf_files), 
                           desc="Extracting PDFs"):
            filename = future_to_file[future]
            try:
                filename, text, resume_data = future.result()
                pdf_texts[filename] = text
                resume_data_dict[filename] = resume_data
            except Exception as e:
                logging.error(f"Unexpected error processing {filename}: {e}")
                pdf_texts[filename] = f"[CRITICAL ERROR: {e}]"
                resume_data_dict[filename] = ResumeData(filename=filename, full_text=f"[CRITICAL ERROR: {e}]")
    
    total_time = time.time() - start_time
    logging.info(f"Completed extraction of {total_files} PDF files in {total_time:.2f} seconds")
    
    # Report success rate
    error_count = sum(1 for text in pdf_texts.values() if text.startswith('[ERROR') or text.startswith('[CRITICAL'))
    if error_count > 0:
        logging.warning(f"Extraction completed with {error_count}/{total_files} errors ({error_count/total_files*100:.1f}%)")
    else:
        logging.info(f"Successfully extracted all {total_files} resumes")
    
    return pdf_texts, resume_data_dict

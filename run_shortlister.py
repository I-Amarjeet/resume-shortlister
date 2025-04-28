import os
import sys
import asyncio
# Add the app directory to the sys.path if it's not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(current_dir, "app")
if os.path.exists(app_dir) and app_dir not in sys.path:
    sys.path.append(app_dir)

# Handle imports to work in both scenarios
try:
    # When running from project root
    from app.resume_processing import extract_text_from_pdfs
    from app.llm_shortlister import shortlist_resumes, EVAL_WEIGHTS
except ImportError:
    # When running from the same directory as the app
    from resume_processing import extract_text_from_pdfs
    from llm_shortlister import shortlist_resumes, EVAL_WEIGHTS

from tqdm import tqdm
from datetime import datetime
import time
import concurrent.futures
import argparse
import signal
import logging
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("resume_shortlister.log"),
        logging.StreamHandler()
    ]
)

# Global variable to track if the process was interrupted
interrupted = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global interrupted
    print("\n\nProcess interrupted! Finishing current batch and saving partial results...")
    interrupted = True
    # Don't exit immediately - let the program save partial results

signal.signal(signal.SIGINT, signal_handler)

def notify_mac(title, message):
    """Send macOS notification"""
    try:
        os.system(f'''osascript -e 'display notification "{message}" with title "{title}"' ''')
        return True
    except Exception as e:
        logging.error(f"Notification failed: {e}")
        return False

def parse_weights(weights_str):
    """Parse comma-separated weights string into a dictionary."""
    if not weights_str:
        return None
        
    weights = {}
    pairs = weights_str.split(',')
    
    for pair in pairs:
        if ':' not in pair:
            continue
        key, value = pair.split(':')
        key = key.strip()
        try:
            value = float(value.strip())
            weights[key] = value
        except ValueError:
            logging.warning(f"Invalid weight value for {key}: {value}")
    
    # Validate weights
    total = sum(weights.values())
    if abs(total - 1.0) > 0.001:  # Allow small rounding errors
        logging.warning(f"Weights do not sum to 1.0 (total: {total}). Normalizing...")
        for key in weights:
            weights[key] = weights[key] / total
            
    return weights

async def process_folder(pdf_dir, output_md, weights=None, max_workers=None, job_description=None):
    """Process a folder of PDFs with batching and progress reporting."""
    global interrupted
    
    try:
        logging.info(f"Starting resume shortlisting process")
        logging.info(f"Input folder: {pdf_dir}")
        logging.info(f"Output file: {output_md}")
        
        if weights:
            logging.info(f"Using custom evaluation weights: {weights}")
        else:
            logging.info(f"Using default evaluation weights: {EVAL_WEIGHTS}")
        
        if job_description:
            logging.info(f"Using job description: {job_description}")
        
        # Extract text and structured data from all PDFs
        start_time = time.time()
        pdf_texts, resume_data_dict = extract_text_from_pdfs(pdf_dir, max_workers=max_workers)
        num_files = len(pdf_texts)
        
        if num_files == 0:
            logging.error("No PDF files found in the directory.")
            return False
        
        extraction_time = time.time() - start_time
        logging.info(f"Extracted text and data from {num_files} PDF files in {extraction_time:.2f} seconds.")
        
        # Check for extraction errors
        error_count = sum(1 for text in pdf_texts.values() if text.startswith('[ERROR') or text.startswith('[CRITICAL'))
        if error_count > 0:
            logging.warning(f"{error_count} PDFs had extraction errors ({error_count/num_files*100:.1f}%)")
        
        # Process PDFs with the LLM
        logging.info(f"Starting LLM shortlisting with batching...")
        shortlist_start = time.time()
        
        # Check if interrupted before starting LLM processing
        if interrupted:
            logging.warning("Process was interrupted before LLM processing began")
            return False
        
        try:
            # Pass both the raw text and structured data to the shortlister
            shortlist_md = await shortlist_resumes(pdf_texts, resume_data_dict, weights, job_description)
            shortlist_time = time.time() - shortlist_start
            
            # Save the results
            with open(output_md, "w", encoding="utf-8") as f:
                f.write(shortlist_md)
            
            total_time = time.time() - start_time
            logging.info(f"Shortlisting complete!")
            logging.info(f"Processed {num_files} resumes in {total_time:.2f} seconds")
            logging.info(f"Results saved to: {output_md}")
            
            # Send notification
            notify_mac("Resume Shortlister", f"Shortlisting complete! Processed {num_files} resumes in {total_time:.1f}s. Results saved.")
            return True
            
        except Exception as e:
            logging.error(f"An error occurred during shortlisting: {str(e)}")
            # Save any partial results if available
            if 'shortlist_md' in locals() and shortlist_md:
                error_output = f"{output_md}.partial"
                with open(error_output, "w", encoding="utf-8") as f:
                    f.write(shortlist_md)
                logging.info(f"Partial results saved to: {error_output}")
            
            notify_mac("Resume Shortlister", f"Error during shortlisting: {str(e)}")
            return False
    
    except Exception as e:
        logging.error(f"Unexpected error in process_folder: {str(e)}")
        notify_mac("Resume Shortlister", f"Critical error: {str(e)}")
        return False

async def main():
    """Command-line interface for the resume shortlister."""
    parser = argparse.ArgumentParser(
        description='Resume Shortlister: Analyze and rank resumes based on custom job requirements.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings for AI Engineering intern role
  python run_shortlister.py /path/to/resumes
  
  # Specify custom output file
  python run_shortlister.py /path/to/resumes -o candidates.md
  
  # Use custom evaluation weights
  python run_shortlister.py /path/to/resumes -w "technical_expertise:0.4,project_complexity:0.3,work_ethic:0.2,certifications:0.1"
  
  # Provide custom job description
  python run_shortlister.py /path/to/resumes -j "job_description.txt"
  
  # Control parallel processing
  python run_shortlister.py /path/to/resumes -t 8
        """
    )
    parser.add_argument('pdf_folder', nargs='?',
                        default=None,
                        help='Folder containing PDF resumes to analyze (default: None)')
    parser.add_argument('-o', '--output', 
                        help='Output markdown file (default: shortlist_YYYYMMDD_HHMMSS.md)')
    parser.add_argument('-w', '--weights', 
                        help='Custom evaluation weights as comma-separated key:value pairs (e.g., "technical_expertise:0.4,project_complexity:0.3")')
    parser.add_argument('-j', '--job_description',
                        help='Path to a text file containing the job description, or the job description itself')
    parser.add_argument('-t', '--threads', type=int, default=None, 
                        help='Maximum number of worker threads for PDF extraction (default: auto)')
    
    args = parser.parse_args()
    
    pdf_dir = args.pdf_folder
    if not os.path.isdir(pdf_dir):
        logging.error(f"Folder not found: {pdf_dir}")
        logging.info(f"Please make sure the resume directory exists or provide a valid path")
        return False
    
    # Set up the output directory
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # If output path is provided, use it as is; otherwise create a timestamped file in the output directory
    if args.output:
        # If a relative path was provided, make it relative to the output directory
        if not os.path.isabs(args.output):
            output_md = os.path.join(output_dir, args.output)
        else:
            output_md = args.output
    else:
        # Create a timestamped file in the output directory
        output_md = os.path.join(output_dir, f"shortlist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    
    # Parse weights if provided
    weights = parse_weights(args.weights) if args.weights else None
    
    # Get job description if provided
    job_description = None
    if args.job_description:
        # Check if it's a file path or the actual description
        if os.path.isfile(args.job_description):
            try:
                with open(args.job_description, 'r', encoding='utf-8') as f:
                    job_description = f.read()
                logging.info(f"Loaded job description from file: {args.job_description}")
            except Exception as e:
                logging.error(f"Error reading job description file: {e}")
                return False
        else:
            # Assume it's the actual job description text
            job_description = args.job_description
            logging.info("Using provided job description text")
    
    return await process_folder(pdf_dir, output_md, weights=weights, max_workers=args.threads, job_description=job_description)

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        # This shouldn't be reached due to signal handler, but just in case
        print("\n[INFO] Process interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        notify_mac("Resume Shortlister", f"Critical error: {str(e)}")
        sys.exit(1)

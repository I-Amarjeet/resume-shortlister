import os
import logging
from openai import AzureOpenAI
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Any, Optional
import asyncio
import time
from tqdm import tqdm
import random
import json
import re
from dataclasses import dataclass, field
from .resume_processing import ResumeData

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_shortlisting.log"),
        logging.StreamHandler()
    ]
)

# Load Azure OpenAI credentials from .env
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")


# Configure the Azure OpenAI client according to the Microsoft guide
client = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT,
                    api_version=AZURE_OPENAI_API_VERSION,
                    api_key=AZURE_OPENAI_KEY)

# Define evaluation weights
EVAL_WEIGHTS = {
    "technical_expertise": 0.35,  # 35% weight for technical skills
    "project_complexity": 0.15,   # 15% weight for project complexity
    "hackathon_experience": 0.25, # 25% for hackathon experience
    "work_ethic": 0.10,           # 10% for work ethic indicators
    "extracurricular": 0.10,      # 10% for relevant activities
    "certifications": 0.5        # 5% for certifications
}

@dataclass
class CandidateEvaluation:
    """Evaluation result for a candidate."""
    name: str = ""
    filename: str = ""
    contact: str = ""
    email: str = ""
    github: str = ""
    linkedin: str = ""
    technical_score: float = 0.0
    project_score: float = 0.0
    hackathon_score: float = 0.0
    work_ethic_score: float = 0.0
    extracurricular_score: float = 0.0
    certification_score: float = 0.0
    overall_score: float = 0.0
    ranking: int = 0
    technical_assessment: List[str] = field(default_factory=list)
    project_complexity: str = ""
    project_details: List[str] = field(default_factory=list)
    hackathon_experience: List[str] = field(default_factory=list)
    work_ethic_indicators: List[str] = field(default_factory=list)
    extracurricular_activities: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    remarks: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "filename": self.filename,
            "contact": self.contact,
            "email": self.email,
            "github": self.github,
            "linkedin": self.linkedin,
            "scores": {
                "technical": self.technical_score,
                "project": self.project_score,
                "hackathon": self.hackathon_score,
                "work_ethic": self.work_ethic_score,
                "extracurricular": self.extracurricular_score,
                "certification": self.certification_score,
                "overall": self.overall_score
            },
            "ranking": self.ranking,
            "details": {
                "technical_assessment": self.technical_assessment,
                "project_complexity": self.project_complexity,
                "project_details": self.project_details,
                "hackathon_experience": self.hackathon_experience,
                "work_ethic_indicators": self.work_ethic_indicators,
                "extracurricular_activities": self.extracurricular_activities,
                "certifications": self.certifications,
                "remarks": self.remarks
            }
        }

    def to_markdown(self) -> str:
        """Convert the evaluation to markdown format."""
        md = f"## {self.ranking}. {self.name}\n"
        md += f"**Score: {self.overall_score:.2f}/5.0**\n\n"

        contact_info = []
        if self.contact:
            contact_info.append(f"Contact: {self.contact}")
        if self.email:
            contact_info.append(f"Email: {self.email}")
        if self.github:
            contact_info.append(f"GitHub: {self.github}")
        if self.linkedin:
            contact_info.append(f"LinkedIn: {self.linkedin}")
        
        if contact_info:
            md += f"*{' | '.join(contact_info)}*\n\n"

        # Strengths and weaknesses section
        md += "### Key Scores\n"
        # Create visual bars for scores
        for category, score in [
            ("Technical Expertise", self.technical_score),
            ("Project Complexity", self.project_score),
            ("Hackathon Experience", self.hackathon_score),
            ("Work Ethic", self.work_ethic_score),
            ("Extracurricular Activities", self.extracurricular_score),
            ("Certifications", self.certification_score)
        ]:
            # Create a visual bar based on the score (0-5)
            bar = "█" * int(score)
            empty = "░" * (5 - int(score))
            md += f"- **{category}:** {bar}{empty} {score:.1f}/5.0\n"

        # Only include sections with content
        if self.technical_assessment:
            md += "\n### Technical Skills\n"
            for item in self.technical_assessment:
                md += f"- {item}\n"

        if self.project_details:
            md += f"\n### Projects ({self.project_complexity})\n"
            for item in self.project_details:
                md += f"- {item}\n"

        if self.hackathon_experience:
            md += "\n### Hackathon Experience\n"
            for item in self.hackathon_experience:
                md += f"- {item}\n"

        if self.work_ethic_indicators:
            md += "\n### Work Ethic\n"
            for item in self.work_ethic_indicators:
                md += f"- {item}\n"

        if self.extracurricular_activities:
            md += "\n### Extracurricular\n"
            for item in self.extracurricular_activities:
                md += f"- {item}\n"

        if self.certifications:
            md += "\n### Certifications\n"
            for item in self.certifications:
                md += f"- {item}\n"

        if self.remarks:
            md += f"\n### Summary\n{self.remarks}\n"

        return md

# Enhanced prompt template based on detailed requirements
EVAL_FRAMEWORK = """
Evaluation Framework for AI Engineering Intern Candidates

When evaluating each resume, carefully assess these criteria and provide a score from 0-5 for each:

1. Technical Expertise (Score 0-5):
   - Natural Language Processing implementations
   - LLM applications including RAG systems
   - Vector database experience
   - Chatbot development
   - Python programming with web frameworks (FastAPI/Flask)
   - Prompt engineering techniques

2. Project Complexity (Score 0-5):
   - Implementation of sophisticated AI models
   - Projects involving multiple technologies (Python, APIs, databases)
   - Projects with quantifiable results (e.g., "increased accuracy by 15%")
   - Clear explanation of technical challenges overcome

3. Hackathon Experience (Score 0-5):
   - Participation in AI/ML hackathons
   - Real-world problem-solving applications

4. Work Ethic and Passion (Score 0-5):
   - Consistent project completion
   - Self-initiated projects
   - Independent learning of new technologies
   - Progressive skill development

5. Extracurricular Activities (Score 0-5):
   - AI/ML competitions
   - Open-source contributions
   - Technical community membership
   - Leadership in tech-related groups

6. Certifications (Score 0-5):
   - Machine Learning/AI specializations
   - Python programming
   - Cloud platforms with ML focus
   - Data Science foundations

For each category, provide a numerical score AND detailed justification.

The overall candidate score should be calculated by weighting the scores as follows:
- Technical Expertise: 35%
- Project Complexity: 25% 
- Hackathon Experience: 10%
- Work Ethic and Passion: 15%
- Extracurricular Activities: 10%
- Certifications: 5%

Return your analysis in the following JSON format:
{
  "name": "Candidate's full name",
  "contact": "Phone number if available",
  "email": "Email if available",
  "github": "GitHub URL if available",
  "linkedin": "LinkedIn URL if available",
  "scores": {
    "technical": 0-5 score,
    "project": 0-5 score,
    "hackathon": 0-5 score,
    "work_ethic": 0-5 score,
    "extracurricular": 0-5 score,
    "certification": 0-5 score,
    "overall": "Weighted average based on specified weights"
  },
  "details": {
    "technical_assessment": ["Bullet points describing technical skills"],
    "project_complexity": "High/Medium/Low",
    "project_details": ["Bullet points describing notable projects"],
    "hackathon_experience": ["Bullet points describing hackathon participation"],
    "work_ethic_indicators": ["Bullet points showing evidence of self-motivation"],
    "extracurricular_activities": ["Bullet points listing relevant activities"],
    "certifications": ["Bullet points listing relevant certifications"],
    "remarks": "Overall assessment of candidate fit"
  }
}
"""

# Improved batching utility with dynamic token allocation
def batch_resumes(resume_data_dict: Dict[str, ResumeData], max_tokens_per_batch=12000, min_batch_size=5) -> List[List[Tuple[str, ResumeData]]]:
    """
    Batch resumes so that each batch fits within the LLM context window.
    Returns a list of lists of (filename, resume_data) tuples.
    
    Args:
        resume_data_dict: Dictionary mapping filenames to resume data
        max_tokens_per_batch: Maximum tokens per batch
        min_batch_size: Minimum number of resumes per batch (will override token limit if needed)
    """
    batches = []
    current_batch = []
    current_tokens = 0

    # Sort resumes by size to better distribute them across batches
    sorted_resumes = sorted(resume_data_dict.items(), key=lambda x: len(x[1].full_text))

    # Calculate prompt token overhead (approximate)
    prompt_overhead = len(EVAL_FRAMEWORK) // 3  # ~1 token per 3 chars
    available_tokens = max_tokens_per_batch - prompt_overhead

    for fname, resume_data in sorted_resumes:
        # More accurate token estimate for mixed content
        tokens = max(1, len(resume_data.full_text) // 3.5)

        # Check if adding this resume would exceed token limit
        if current_batch and current_tokens + tokens > available_tokens and len(current_batch) >= min_batch_size:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append((fname, resume_data))
        current_tokens += tokens

    if current_batch:
        batches.append(current_batch)

    logging.info(f"Created {len(batches)} batches from {len(resume_data_dict)} resumes")
    for i, batch in enumerate(batches):
        logging.info(f"  Batch {i+1}: {len(batch)} resumes, ~{sum(max(1, len(data[1].full_text) // 3.5) for data in batch)} tokens")
    return batches

def analyze_batch(batch: List[Tuple[str, ResumeData]], eval_weights: Dict[str, float] = None, attempt=0, job_description=None, eval_framework=None) -> List[Dict[str, Any]]:
    """
    Send a batch of resumes to Azure OpenAI and return the LLM's evaluations.
    Includes retry logic for handling rate limits and other errors.
    
    Args:
        batch: List of tuples containing filename and resume data
        eval_weights: Optional custom weights for evaluation criteria
        attempt: Current attempt number for retry logic
        job_description: Optional custom job description
        eval_framework: Optional custom evaluation framework
    """
    max_attempts = 5
    backoff_time = 2

    # Use default weights if none provided
    weights = eval_weights if eval_weights else EVAL_WEIGHTS
    
    # Check for custom evaluation criteria
    custom_criteria = {}
    standard_criteria = {"technical_expertise", "project_complexity", "hackathon_experience", 
                        "work_ethic", "extracurricular", "certifications"}
    
    for key in weights.keys():
        if key not in standard_criteria:
            custom_criteria[key] = weights[key]

    # Prepare resumes data and prompt
    resumes_data = []
    for _, resume_data in batch:
        resume_text = {}

        # Add structured data if available
        resume_text["filename"] = resume_data.filename
        resume_text["name"] = resume_data.name if resume_data.name else "Unknown"
        resume_text["email"] = resume_data.email if resume_data.email else ""
        resume_text["phone"] = resume_data.phone if resume_data.phone else ""
        resume_text["github"] = resume_data.github if resume_data.github else ""
        resume_text["linkedin"] = resume_data.linkedin if resume_data.linkedin else ""

        # Add section data if available
        sections = []
        if resume_data.education:
            sections.append(f"EDUCATION:\n{resume_data.education[0][:1000]}")
        if resume_data.experience:
            sections.append(f"EXPERIENCE:\n{resume_data.experience[0][:1500]}")
        if resume_data.skills:
            sections.append(f"SKILLS:\n{resume_data.skills[0][:1000]}")
        if resume_data.projects:
            sections.append(f"PROJECTS:\n{resume_data.projects[0][:1500]}")
        if resume_data.certifications:
            sections.append(f"CERTIFICATIONS:\n{resume_data.certifications[0][:500]}")

        # If sections were found, use them. Otherwise, use smart truncation of full text
        if sections:
            resume_content = "\n\n".join(sections)
        else:
            # Smart truncation for full text
            full_text = resume_data.full_text
            if len(full_text) <= 10000:
                resume_content = full_text
            else:
                # Take first 6K chars and last 4K chars
                resume_content = f"{full_text[:6000]}...\n\n[Middle content omitted]\n\n{full_text[-4000:]}"

        # Format resume data for prompt
        resumes_data.append({
            "filename": resume_data.filename,
            "name": resume_text["name"],
            "email": resume_text["email"],
            "phone": resume_text["phone"],
            "github": resume_text["github"],
            "linkedin": resume_text["linkedin"],
            "content": resume_content
        })

    # Prepare weights info for the prompt
    weights_info = "\n".join([f"- {key.replace('_', ' ').title()}: {value*100:.1f}%" for key, value in weights.items()])

    # Use the provided evaluation framework if available, otherwise use the default
    framework_to_use = eval_framework if eval_framework else EVAL_FRAMEWORK
    
    # If there are custom criteria, modify the evaluation framework to include them
    if custom_criteria and not eval_framework:
        # Get the base framework
        modified_framework = framework_to_use
        
        # Add custom criteria to the evaluation framework
        custom_section = "\n\nADDITIONAL EVALUATION CRITERIA:\n"
        for i, (key, value) in enumerate(custom_criteria.items(), start=7):
            criterion_name = key.replace('_', ' ').title()
            custom_section += f"\n{i}. {criterion_name} (Score 0-5):\n"
            
            # Look for description if available in the weights dictionary
            if f"{key}_description" in weights:
                description = weights[f"{key}_description"]
                custom_section += f"   - {description}\n"
            
        # Add the custom criteria to the framework
        modified_framework += custom_section
        
        # Update JSON return format in the prompt
        scores_section = '"scores": {\n    "technical": 0-5 score,\n    "project": 0-5 score,\n    "hackathon": 0-5 score,\n    "work_ethic": 0-5 score,\n    "extracurricular": 0-5 score,\n    "certification": 0-5 score,'
        
        # Add custom criteria to the scores section
        for key in custom_criteria.keys():
            key_short = key.split('_')[0] if '_' in key else key  # Simplify key for the JSON
            scores_section += f'\n    "{key_short}": 0-5 score,'
            
        scores_section += '\n    "overall": "Weighted average based on specified weights"\n  },'
        
        # Replace the scores section in the framework
        modified_framework = modified_framework.replace('"scores": {\n    "technical": 0-5 score,\n    "project": 0-5 score,\n    "hackathon": 0-5 score,\n    "work_ethic": 0-5 score,\n    "extracurricular": 0-5 score,\n    "certification": 0-5 score,\n    "overall": "Weighted average based on specified weights"\n  },', scores_section)
        
        framework_to_use = modified_framework

    # Build the prompt
    prompt = f"{framework_to_use}\n\nEvaluation weights:\n{weights_info}\n\n"
    prompt += "Please evaluate each resume and return a JSON object for each candidate. Here are the resumes:\n\n"

    for resume in resumes_data:
        prompt += f"==== RESUME: {resume['filename']} ====\n"
        if resume['name']: prompt += f"Name: {resume['name']}\n"
        if resume['email']: prompt += f"Email: {resume['email']}\n"
        if resume['phone']: prompt += f"Phone: {resume['phone']}\n"
        if resume['github']: prompt += f"GitHub: {resume['github']}\n"
        if resume['linkedin']: prompt += f"LinkedIn: {resume['linkedin']}\n\n"
        prompt += f"{resume['content']}\n\n"
        prompt += "==== END OF RESUME ====\n\n"

    try:
        # Using the OpenAI library as per Microsoft Learning guide
        response = client.chat.completions.create(model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are an expert technical recruiter specializing in AI engineering talent evaluation."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=4096,
        presence_penalty=0.1,
        frequency_penalty=0.1,
        response_format={"type": "json_object"})

        content = response.choices[0].message.content

        # Extract JSON objects from the response
        try:
            result = json.loads(content)
            if "candidates" in result:
                return result["candidates"]
            elif isinstance(result, list):
                return result
            else:
                # Single candidate
                return [result]
        except json.JSONDecodeError:
            # Try to extract JSON objects using regex if the response isn't valid JSON
            json_objects = []
            json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}))*\}'
            matches = re.findall(json_pattern, content)

            for match in matches:
                try:
                    obj = json.loads(match)
                    json_objects.append(obj)
                except:
                    pass

            if json_objects:
                return json_objects

            raise ValueError(f"Failed to parse JSON from response: {content[:100]}...")

    except Exception as e:
        if attempt < max_attempts:
            # Exponential backoff with jitter
            sleep_time = (backoff_time ** attempt) + random.uniform(0, 1)
            print(f"Error in batch analysis, retrying in {sleep_time:.2f}s: {str(e)}")
            time.sleep(sleep_time)
            return analyze_batch(batch, eval_weights, attempt + 1, job_description, eval_framework)
        else:
            # If all retries fail, return error message
            print(f"Failed after {max_attempts} attempts: {str(e)}")
            return [{"error": f"Failed to analyze resumes: {[data[0] for data in batch]}"} for _ in batch]

def extract_candidate_evaluations(batch_results: List[Dict[str, Any]], batch: List[Tuple[str, ResumeData]]) -> List[CandidateEvaluation]:
    """Extract structured candidate evaluations from LLM output."""
    evaluations = []

    # Map filenames to resume data for lookup
    filename_map = {data[0]: data[1] for data in batch}

    for result in batch_results:
        if "error" in result:
            # Handle error case
            eval_obj = CandidateEvaluation()
            if len(batch) == 1:
                eval_obj.filename = batch[0][0]
                eval_obj.name = batch[0][1].name
            eval_obj.remarks = f"Error: {result['error']}"
            evaluations.append(eval_obj)
            continue

        try:
            eval_obj = CandidateEvaluation()

            # Basic info
            eval_obj.name = result.get("name", "")

            # Try to match with filename
            matching_files = []
            for filename in filename_map.keys():
                if eval_obj.name and eval_obj.name.lower() in filename.lower():
                    matching_files.append(filename)

            if matching_files:
                eval_obj.filename = matching_files[0]
            elif len(batch) == 1:
                # If only one resume in batch, use that filename
                eval_obj.filename = batch[0][0]

            # Contact info
            eval_obj.contact = result.get("contact", "")
            eval_obj.email = result.get("email", "")
            eval_obj.github = result.get("github", "")
            eval_obj.linkedin = result.get("linkedin", "")

            # Fill in structured data from resume_data if missing
            if eval_obj.filename in filename_map:
                resume_data = filename_map[eval_obj.filename]
                if not eval_obj.name and resume_data.name:
                    eval_obj.name = resume_data.name
                if not eval_obj.email and resume_data.email:
                    eval_obj.email = resume_data.email
                if not eval_obj.contact and resume_data.phone:
                    eval_obj.contact = resume_data.phone
                if not eval_obj.github and resume_data.github:
                    eval_obj.github = resume_data.github
                if not eval_obj.linkedin and resume_data.linkedin:
                    eval_obj.linkedin = resume_data.linkedin

            # Scores
            scores = result.get("scores", {})
            eval_obj.technical_score = float(scores.get("technical", 0))
            eval_obj.project_score = float(scores.get("project", 0))
            eval_obj.hackathon_score = float(scores.get("hackathon", 0))
            eval_obj.work_ethic_score = float(scores.get("work_ethic", 0))
            eval_obj.extracurricular_score = float(scores.get("extracurricular", 0))
            eval_obj.certification_score = float(scores.get("certification", 0))

            # Overall score - either use provided or calculate from weights
            if "overall" in scores:
                eval_obj.overall_score = float(scores.get("overall", 0))
            else:
                # Calculate using default weights
                eval_obj.overall_score = (
                    eval_obj.technical_score * EVAL_WEIGHTS["technical_expertise"] +
                    eval_obj.project_score * EVAL_WEIGHTS["project_complexity"] +
                    eval_obj.hackathon_score * EVAL_WEIGHTS["hackathon_experience"] +
                    eval_obj.work_ethic_score * EVAL_WEIGHTS["work_ethic"] +
                    eval_obj.extracurricular_score * EVAL_WEIGHTS["extracurricular"] +
                    eval_obj.certification_score * EVAL_WEIGHTS["certifications"]
                )

            # Details
            details = result.get("details", {})
            eval_obj.technical_assessment = details.get("technical_assessment", [])
            eval_obj.project_complexity = details.get("project_complexity", "")
            eval_obj.project_details = details.get("project_details", [])
            eval_obj.hackathon_experience = details.get("hackathon_experience", [])
            eval_obj.work_ethic_indicators = details.get("work_ethic_indicators", [])
            eval_obj.extracurricular_activities = details.get("extracurricular_activities", [])
            eval_obj.certifications = details.get("certifications", [])
            eval_obj.remarks = details.get("remarks", "")

            evaluations.append(eval_obj)

        except Exception as e:
            print(f"Error extracting evaluation: {str(e)}")
            print(f"Problematic result: {result}")
            # Add a minimal evaluation as fallback
            eval_obj = CandidateEvaluation()
            if len(batch) == 1:
                eval_obj.filename = batch[0][0]
            eval_obj.remarks = f"Error extracting evaluation: {str(e)}"
            evaluations.append(eval_obj)

    return evaluations

def generate_candidate_summary(evaluations: List[CandidateEvaluation]) -> str:
    """Generate an executive summary of all candidates."""
    if not evaluations:
        return "# Resume Shortlister Results\n\nNo candidates were evaluated. Please check input directory."

    # Count candidates by score ranges
    ranges = {
        "outstanding": 0,  # 4.5+
        "excellent": 0,    # 4.0-4.49
        "good": 0,         # 3.5-3.99
        "average": 0,      # 3.0-3.49
        "below_average": 0 # <3.0
    }

    for eval_obj in evaluations:
        if eval_obj.overall_score >= 4.5:
            ranges["outstanding"] += 1
        elif eval_obj.overall_score >= 4.0:
            ranges["excellent"] += 1
        elif eval_obj.overall_score >= 3.5:
            ranges["good"] += 1
        elif eval_obj.overall_score >= 3.0:
            ranges["average"] += 1
        else:
            ranges["below_average"] += 1

    # Get top candidates
    top_candidates = sorted(evaluations, key=lambda x: x.overall_score, reverse=True)
    top_10 = top_candidates[:10]  # Show top 10 instead of just 5

    # Calculate average scores per category
    avg_technical = sum(e.technical_score for e in evaluations) / len(evaluations)
    avg_project = sum(e.project_score for e in evaluations) / len(evaluations)
    avg_hackathon = sum(e.hackathon_score for e in evaluations) / len(evaluations)
    avg_work_ethic = sum(e.work_ethic_score for e in evaluations) / len(evaluations)
    avg_extracurricular = sum(e.extracurricular_score for e in evaluations) / len(evaluations)
    avg_certification = sum(e.certification_score for e in evaluations) / len(evaluations)

    # Build the summary
    summary = "# Resume Shortlister Results\n\n"
    summary += f"## Executive Summary\n\n"
    summary += f"### Candidate Pool Overview\n\n"
    summary += f"**Total candidates evaluated:** {len(evaluations)}\n\n"
    
    # Create a visual chart for score distribution
    summary += f"### Score Distribution\n\n"
    summary += "| Category | Count | Visualization |\n"
    summary += "|----------|------:|---------------|\n"
    summary += f"| Outstanding (4.5+) | {ranges['outstanding']} | {'█' * ranges['outstanding']} |\n"
    summary += f"| Excellent (4.0-4.49) | {ranges['excellent']} | {'█' * ranges['excellent']} |\n"
    summary += f"| Good (3.5-3.99) | {ranges['good']} | {'█' * ranges['good']} |\n"
    summary += f"| Average (3.0-3.49) | {ranges['average']} | {'█' * ranges['average']} |\n"
    summary += f"| Below Average (<3.0) | {ranges['below_average']} | {'█' * ranges['below_average']} |\n\n"

    # Create a visual table for category averages
    summary += f"### Average Scores by Category\n\n"
    summary += "| Category | Score | Rating |\n"
    summary += "|----------|------:|--------|\n"
    
    for name, score in [
        ("Technical Expertise", avg_technical),
        ("Project Complexity", avg_project),
        ("Hackathon Experience", avg_hackathon),
        ("Work Ethic", avg_work_ethic),
        ("Extracurricular Activities", avg_extracurricular),
        ("Certifications", avg_certification)
    ]:
        # Create a rating visualization
        rating = "█" * int(score) + "░" * (5 - int(score))
        summary += f"| {name} | {score:.2f} | {rating} |\n"
    
    summary += "\n### Top Candidates\n\n"
    summary += "| Rank | Name | Score | Key Strength |\n"
    summary += "|-----:|------|------:|-------------|\n"
    
    for i, candidate in enumerate(top_10, 1):
        name = candidate.name or f"Candidate {candidate.filename}"
        # Truncate name if too long
        if len(name) > 30:
            name = name[:27] + "..."
            
        key_strength = candidate.remarks.split('.')[0] if candidate.remarks else 'N/A'
        # Truncate strength if too long
        if len(key_strength) > 80:
            key_strength = key_strength[:77] + "..."
            
        summary += f"| {i} | {name} | {candidate.overall_score:.2f} | {key_strength} |\n"

    summary += "\n---\n\n"
    return summary

def generate_candidate_summary_table(evaluations: List[CandidateEvaluation]) -> str:
    """
    Generate a summary table of all candidates with their contact information.
    This will be added at the end of the report.
    """
    if not evaluations:
        return "## Candidate Contact Information\n\nNo candidates were evaluated."
    
    # Sort by overall score (highest first)
    sorted_candidates = sorted(evaluations, key=lambda x: x.overall_score, reverse=True)
    
    # Create the table header
    table = "## Candidate Contact Information\n\n"
    table += "| Rank | Name | Phone | Email | GitHub | LinkedIn | Score |\n"
    table += "|-----:|------|-------|-------|--------|----------|------:|\n"
    
    # Add each candidate to the table
    for eval_obj in sorted_candidates:
        name = eval_obj.name or f"Candidate {eval_obj.filename}"
        phone = eval_obj.contact or "—"
        email = eval_obj.email or "—"
        github = eval_obj.github or "—"
        linkedin = eval_obj.linkedin or "—"
        
        table += f"| {eval_obj.ranking} | {name} | {phone} | {email} | {github} | {linkedin} | {eval_obj.overall_score:.2f} |\n"
    
    return table

def customize_eval_framework(job_description: str) -> str:
    """
    Create a customized evaluation framework based on the provided job description.
    This allows the system to adapt to different roles and requirements.
    """
    # First, extract key requirements and skills from the job description using Azure OpenAI
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are an expert recruiter who can identify key skills and requirements from job descriptions."},
                {"role": "user", "content": f"""
                Analyze the following job description and extract:
                1. Key technical skills required (up to 10)
                2. Required education level
                3. Required experience level
                4. Key soft skills required
                5. Main responsibilities of the role
                
                Format your response as a structured JSON object with these categories.
                
                Job Description:
                {job_description}
                """}
            ],
            temperature=0.3,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        # Extract the structured analysis
        job_analysis = json.loads(response.choices[0].message.content)
        
        # Now use this analysis to customize the evaluation framework
        custom_framework = f"""
        Evaluation Framework for Candidates

        When evaluating each resume for the following position, carefully assess these criteria and provide a score from 0-5 for each:

        JOB DETAILS:
        {job_description}

        1. Technical Expertise (Score 0-5):
        """
        
        # Add technical skills from the analysis
        if "technical_skills" in job_analysis and job_analysis["technical_skills"]:
            for skill in job_analysis["technical_skills"]:
                custom_framework += f"   - {skill}\n"
        else:
            custom_framework += "   - Relevant technical skills for the position\n"
            
        # Continue with other criteria
        custom_framework += """
        2. Project Complexity (Score 0-5):
           - Implementation of relevant projects
           - Projects involving multiple technologies
           - Projects with quantifiable results
           - Clear explanation of technical challenges overcome

        3. Relevant Experience (Score 0-5):
           - Previous roles in similar positions
           - Industry experience
           - Relevant achievements
        
        4. Work Ethic and Passion (Score 0-5):
           - Consistent project completion
           - Self-initiated projects
           - Independent learning of new technologies
           - Progressive skill development
        
        5. Education and Training (Score 0-5):
           - Relevant degrees or certifications
           - Continuous learning and development
           - Industry-relevant courses
        
        6. Communication and Collaboration (Score 0-5):
           - Evidence of teamwork
           - Communication skills
           - Leadership experience

        For each category, provide a numerical score AND detailed justification.

        Return your analysis in the following JSON format:
        {
          "name": "Candidate's full name",
          "contact": "Phone number if available",
          "email": "Email if available",
          "github": "GitHub URL if available",
          "linkedin": "LinkedIn URL if available",
          "scores": {
            "technical": 0-5 score,
            "project": 0-5 score,
            "hackathon": 0-5 score,
            "work_ethic": 0-5 score,
            "extracurricular": 0-5 score,
            "certification": 0-5 score,
            "overall": "Weighted average based on specified weights"
          },
          "details": {
            "technical_assessment": ["Bullet points describing technical skills"],
            "project_complexity": "High/Medium/Low",
            "project_details": ["Bullet points describing notable projects"],
            "hackathon_experience": ["Bullet points describing hackathon participation"],
            "work_ethic_indicators": ["Bullet points showing evidence of self-motivation"],
            "extracurricular_activities": ["Bullet points listing relevant activities"],
            "certifications": ["Bullet points listing relevant certifications"],
            "remarks": "Overall assessment of candidate fit"
          }
        }
        """
        
        logging.info("Successfully created custom evaluation framework based on job description")
        return custom_framework
        
    except Exception as e:
        logging.error(f"Error creating custom evaluation framework: {e}")
        # Fall back to the default framework
        logging.info("Falling back to default evaluation framework")
        return EVAL_FRAMEWORK

async def shortlist_resumes(resume_texts: Dict[str, str], resume_data_dict: Dict[str, ResumeData], eval_weights: Dict[str, float] = None, job_description: str = None) -> str:
    """
    Batch resumes, analyze each batch, and merge Markdown results.
    Now includes structured data extraction and executive summary.
    
    Args:
        resume_texts: Dictionary of filename -> resume text
        resume_data_dict: Dictionary of filename -> structured resume data
        eval_weights: Optional custom weights for evaluation criteria
        job_description: Optional custom job description to evaluate candidates against
    """
    # Use default weights if none provided
    weights = eval_weights if eval_weights else EVAL_WEIGHTS

    # If job description is provided, modify the evaluation framework
    eval_framework = EVAL_FRAMEWORK
    if job_description:
        # Create a customized evaluation framework based on the job description
        eval_framework = customize_eval_framework(job_description)
        logging.info(f"Using custom evaluation framework based on provided job description")

    batches = batch_resumes(resume_data_dict)
    all_evaluations = []

    # Dynamic rate limiting
    min_delay = 0.5
    max_delay = 10.0
    current_delay = min_delay
    consecutive_successes = 0
    consecutive_failures = 0

    for i, batch in enumerate(tqdm(batches, desc="Processing batches")):
        batch_start = time.time()

        # Use loop.run_in_executor to run the synchronous analyze_batch in a separate thread
        loop = asyncio.get_event_loop()
        try:
            batch_results = await loop.run_in_executor(
                None, 
                lambda: analyze_batch(batch, weights, job_description=job_description, eval_framework=eval_framework)
            )

            # Extract structured evaluations from results
            batch_evaluations = extract_candidate_evaluations(batch_results, batch)
            all_evaluations.extend(batch_evaluations)

            # Adjust delay based on success
            consecutive_successes += 1
            consecutive_failures = 0
            if consecutive_successes >= 3 and current_delay > min_delay:
                current_delay = max(min_delay, current_delay * 0.8)

            # Report progress
            batch_time = time.time() - batch_start
            print(f"Batch {i+1}/{len(batches)}: Processed {len(batch)} resumes in {batch_time:.2f}s - Next delay: {current_delay:.2f}s")

        except Exception as e:
            consecutive_failures += 1
            consecutive_successes = 0
            current_delay = min(max_delay, current_delay * 2)
            print(f"Batch {i+1}/{len(batches)} failed: {str(e)}")

            # Create error evaluations for this batch
            for filename, resume_data in batch:
                error_eval = CandidateEvaluation(
                    filename=filename,
                    name=resume_data.name,
                    remarks=f"Error processing in batch {i+1}: {str(e)}"
                )
                all_evaluations.append(error_eval)

        # Delay before next batch - only if we have more batches to process
        if i < len(batches) - 1:
            await asyncio.sleep(current_delay)

    # Sort candidates by overall score and assign rankings
    sorted_evaluations = sorted(all_evaluations, key=lambda x: x.overall_score, reverse=True)
    for i, eval_obj in enumerate(sorted_evaluations, 1):
        eval_obj.ranking = i

    # Generate executive summary
    executive_summary = generate_candidate_summary(sorted_evaluations)

    # Generate individual markdown for each candidate
    candidate_markdowns = [eval_obj.to_markdown() for eval_obj in sorted_evaluations]
    
    # Generate candidate summary table
    candidate_table = generate_candidate_summary_table(sorted_evaluations)

    # Combine everything into final result
    final_result = executive_summary
    final_result += "\n\n# Detailed Candidate Evaluations\n\n"
    final_result += "\n\n---\n\n".join(candidate_markdowns)
    final_result += "\n\n---\n\n"
    final_result += candidate_table

    return final_result

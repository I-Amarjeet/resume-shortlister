# Resume Shortlister

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI-0078D4)](https://azure.microsoft.com/en-us/services/cognitive-services/openai/)

A powerful, AI-driven application that automatically analyzes and shortlists resumes for AI engineering positions using Azure OpenAI's GPT models.

## 📋 Overview

Resume Shortlister processes hundreds of PDF resumes, extracting structured information and using a sophisticated evaluation framework to rank candidates according to technical expertise, project complexity, work ethic, and other relevant factors. Save countless hours of manual resume screening with intelligent, consistent analysis.

### Key Features

- **Batch Processing**: Efficiently handles 250+ resumes, with optimized performance for large datasets
- **Intelligent Extraction**: Extracts structured data (name, contact details, education, experience) from PDFs
- **Custom Evaluation Weights**: Configure your own weights for different evaluation criteria
- **Comprehensive Reports**: Generates detailed Markdown reports with executive summary and individual evaluations
- **Web Interface**: Optional FastAPI web application for interactive usage
- **Custom Job Descriptions**: Adapt the evaluation framework to any role with your own job description

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or later
- Azure OpenAI API credentials

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/resume-shortlister.git
cd resume-shortlister
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv resume
source resume/bin/activate  # On Windows: resume\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your Azure OpenAI credentials:
```
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o  # or your specific deployment name
AZURE_OPENAI_API_VERSION=2024-02-15-preview  # or current version
```

## 💻 Usage

### Command Line Interface

The simplest way to run Resume Shortlister is with the CLI:

```bash
# Basic usage with default settings
python run_shortlister.py /path/to/resume/pdfs

# Specify output file
python run_shortlister.py /path/to/resume/pdfs -o shortlisted_candidates.md

# Custom evaluation weights
python run_shortlister.py /path/to/resume/pdfs -w "technical_expertise:0.4,project_complexity:0.3,work_ethic:0.15,hackathon_experience:0.05,extracurricular:0.05,certifications:0.05"

# Provide custom job description (either file path or direct text)
python run_shortlister.py /path/to/resume/pdfs -j "job_description.txt"
# OR
python run_shortlister.py /path/to/resume/pdfs -j "We are looking for an AI Engineer with..."

# Control parallel processing threads
python run_shortlister.py /path/to/resume/pdfs -t 8
```

### Web Interface

For a more interactive experience, you can run the web application:

```bash
cd app
uvicorn main:app --reload
```

Then visit `http://localhost:8000` in your browser.

## 📝 Evaluation Framework

Shortlisting is based on the following criteria:

| Criterion | Default Weight | Description |
|-----------|----------------|-------------|
| Technical Expertise | 35% | NLP, LLM concepts, Python, web frameworks |
| Project Complexity | 25% | Sophistication, technologies used, results |
| Hackathon Experience | 10% | Participation and outcomes |
| Work Ethic | 15% | Self-motivation, initiative, consistent delivery |
| Extracurricular Activities | 10% | Competitions, open-source, communities |
| Certifications | 5% | Relevant technical certifications |

The evaluation criteria can be customized by:
1. Providing custom weights with the `-w` parameter
2. Providing a custom job description with the `-j` parameter, which will adapt the evaluation framework to the specific role requirements

## 📊 Output Format

The generated Markdown report includes:

1. **Executive Summary**:
   - Overall statistics (number of candidates, score distribution)
   - Average scores across all categories
   - Top 10 candidates with key strengths

2. **Detailed Candidate Evaluations**:
   - Contact information (name, email, GitHub, LinkedIn)
   - Scores for each evaluation criterion
   - Technical assessment in bullet points
   - Project complexity rating with details
   - Other criterion-specific details
   - Overall remarks

3. **Contact Information Table**:
   - Summary table with all candidate contact details for easy reference

## 📈 Latest Run Statistics (April 26, 2025)

Our most recent evaluation analyzed 18 candidates for AI engineering internships, with the following results:

### Score Distribution
- Outstanding (4.5+): 1 candidate
- Excellent (4.0-4.49): 1 candidate
- Good (3.5-3.99): 7 candidates
- Average (3.0-3.49): 1 candidate
- Below Average (<3.0): 8 candidates

### Average Scores by Category
- Technical Expertise: 3.22/5.0
- Project Complexity: 3.25/5.0
- Hackathon Experience: 1.19/5.0
- Work Ethic: 3.64/5.0
- Extracurricular Activities: 2.36/5.0
- Certifications: 2.06/5.0

### Sample Candidate Profile

```
## Imaginary Patil
Score: 3.65/5.0

Key Scores:
- Technical Expertise: 4.0/5.0
- Project Complexity: 4.0/5.0
- Hackathon Experience: 2.0/5.0
- Work Ethic: 4.0/5.0
- Extracurricular Activities: 2.0/5.0
- Certifications: 3.0/5.0

Technical Skills:
- Experience with YOLOv11 for pose estimation in fitness tracking
- Implemented CNN for image classification
- Proficient in Python and OpenCV for real-time video processing

Projects:
- Voice-enabled AI fitness tracker using YOLOv11 and OpenCV
- CNN-based image classifier with hyperparameter tuning
- Student information management system
```

## 🔧 Advanced Features

### Custom Job Descriptions

The system can adapt to different hiring needs by analyzing a provided job description. When you pass a job description with the `-j` parameter, the system:

1. Extracts key technical skills, requirements, and responsibilities
2. Creates a custom evaluation framework tailored to the specific role
3. Evaluates candidates based on this customized framework

This makes the tool flexible for hiring across different positions and departments.

### Smart Resume Parsing

The system uses multiple extraction techniques to handle various PDF formats:

1. Primary extraction with PDFPlumber
2. Fallback extraction with PyPDFium2 if the primary extraction fails
3. Structured information extraction for contact details, education, experience, etc.

### Intelligent Batching

For large resume collections, the system:

1. Organizes resumes into optimal batches based on size
2. Processes batches in parallel with adjustable thread count
3. Implements dynamic rate limiting to prevent API throttling

## 🔧 Customization

### Evaluation Weights

Modify the weights in `app/llm_shortlister.py` or pass custom weights via command line:

```python
EVAL_WEIGHTS = {
    "technical_expertise": 0.35,
    "project_complexity": 0.25,
    "hackathon_experience": 0.10,
    "work_ethic": 0.15,
    "extracurricular": 0.10,
    "certifications": 0.05
}
```

### Prompt Customization

The evaluation framework prompt can be customized in `app/llm_shortlister.py`.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please make sure to update tests as appropriate.

## 🙏 Acknowledgments

- [Azure OpenAI](https://azure.microsoft.com/en-us/services/cognitive-services/openai/) for the powerful LLM capabilities
- [PDFPlumber](https://github.com/jsvine/pdfplumber) and [PyPDFium2](https://github.com/pypdfium2-team/pypdfium2) for PDF extraction
- [FastAPI](https://fastapi.tiangolo.com/) for the web interface
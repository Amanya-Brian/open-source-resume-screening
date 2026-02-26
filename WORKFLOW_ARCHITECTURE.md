# Resume Screening System - Complete Workflow Architecture

## Overview
AI-powered multi-agent resume screening system with integrated fairness analysis, LLM-based evaluation, and web portal interface.

---

## Complete End-to-End Workflow

### 1. **Data Synchronization** (Data Fetching Agent)
**Trigger**: User clicks "Sync All Data" or "Sync Applications" button in web portal

**Process**:
1. System fetches job listings from TalentMatch API (`http://localhost:5000/api/job-listings`)
2. Stores job data in MongoDB `job_listings` collection
3. For each job, fetches all applications
4. Stores applications in `applications` collection
5. Parses resumes (PDF/DOCX) and stores in `resumes` collection

**API Endpoint**: `POST /api/sync/all` or `POST /api/sync/jobs/{job_id}/applications`

**Collections Updated**:
- `job_listings`
- `applications`
- `resumes`

---

### 2. **Candidate Screening** (LLM-based Screening Agent)
**Trigger**: User views job details page and clicks "Run Screening" button

**Process**:
1. User opens job details page: `GET /jobs/{job_id}`
2. Portal displays job info, qualifications, applications
3. User clicks "Screen" button
4. Frontend calls: `POST /api/screening/jobs/{job_id}/screen`
5. **ScreeningService.screen_job_candidates()** executes:

   **Step 5a: Candidate Evaluation Loop**
   - For each application:
     - Fetch candidate's resume and cover letter from MongoDB
     - Combine into full text
     - **Try LLM-based evaluation** (if Ollama available):
       - Call `LLMService.evaluate_candidate()`
       - Send candidate text + job requirements to Llama3
       - LLM scores 6 criteria (0-5 scale):
         * Education (15% weight)
         * Experience (25% weight)
         * Technical Skills (20% weight)
         * Industry Knowledge (15% weight)
         * Leadership (15% weight)
         * Communication (10% weight)
       - LLM provides evidence for each score
       - LLM lists strengths and concerns
     - **Fallback to rule-based** (if LLM fails):
       - Use keyword matching and pattern recognition
       - Extract skills using regex
       - Score based on qualification matches
   - Calculate total weighted score (0-5)
   - Calculate percentage (0-100%)
   - Assign recommendation:
     * ≥90% = STRONG YES
     * ≥80% = YES
     * ≥65% = Maybe
     * <65% = No

   **Step 5b: Ranking**
   - Sort all candidates by total weighted score (descending)
   - Assign rank positions (1, 2, 3, ...)

   **Step 5c: Fairness Analysis** ⭐ **NEW - THIS WAS MISSING**
   - Call `_run_fairness_analysis()`:
     - Convert evaluations to `RankedCandidate` format
     - Convert applications to `Student` format (with demographics)
     - Create `FairnessAgent` instance
     - Execute fairness analysis:
       * **Disparate Impact Ratio (DIR)** for protected attributes:
         - Gender, Age Group, Ethnicity, Nationality
         - DIR = P(selected | minority) / P(selected | majority)
         - **Target: ≥ 0.8 (Four-Fifths Rule)**
       * **Counterfactual Analysis**:
         - Sample 20 candidates
         - Modify protected attributes (e.g., change gender)
         - Verify ranking doesn't change
         - **Target: 0% variance**
       * **Demographic Parity**:
         - Measure selection rate equality across groups
     - Generate `FairnessReport`:
       * `is_compliant`: True if DIR ≥ 0.8 AND variance = 0%
       * `violations`: List of fairness violations
       * `recommendations`: Suggested fixes

   **Step 5d: Storage**
   - Store each candidate's evaluation in `screening_results` collection
   - **Store fairness report** in `fairness_reports` collection
   - Log completion

6. **Response** returns to portal with success status
7. Page reloads and displays ranked candidates

**API Endpoints**:
- `POST /api/screening/jobs/{job_id}/screen` - Trigger screening
- `GET /api/data/jobs/{job_id}/results` - Get screening results
- `GET /api/data/jobs/{job_id}/fairness` - Get fairness report

**Collections Updated**:
- `screening_results` (candidate evaluations)
- `fairness_reports` (fairness analysis)

---

### 3. **Results Display** (Web Portal)
**Trigger**: Page reload after screening completes

**Process**:
1. Portal fetches screening results from MongoDB
2. Displays ranked candidates table:
   - Rank badge (gold/silver/bronze for top 3)
   - Candidate name
   - Total score (0-5)
   - Percentage bar (color-coded)
   - Recommendation badge (STRONG YES, YES, Maybe, No)
   - "View" button for details
3. **Fairness report card auto-loads**:
   - Frontend calls: `GET /api/data/jobs/{job_id}/fairness`
   - Displays 3 metric cards:
     * **Compliance Status**: Green checkmark or red warning
     * **Disparate Impact Ratio**: Numeric value with target
     * **Demographic Parity**: Percentage score
   - Shows violations (if any) in red alert box
   - Shows recommendations in blue info box
   - Shows attribute variance table (counterfactual results)

**Pages**:
- `GET /jobs/{job_id}` - Job detail with ranked candidates
- `GET /jobs/{job_id}/candidates/{candidate_id}` - Individual candidate detail

---

### 4. **Candidate Detail View** (Explanation Agent)
**Trigger**: User clicks "View" button for a candidate

**Process**:
1. Portal fetches candidate's full evaluation from MongoDB
2. Displays:
   - Rank badge and overall score
   - Scoring matrix (all 6 criteria with scores, weights, evidence)
   - Strengths list (from LLM or rule-based)
   - Concerns list (from LLM or rule-based)
   - Detailed notes/summary (from LLM)
3. LLM-generated explanation includes:
   - Professional 2-3 sentence summary
   - Key strengths highlighting why candidate fits
   - Areas of concern requiring clarification
   - Suggested interview questions (optional)

**Page**: `GET /jobs/{job_id}/candidates/{candidate_id}`

---

## Data Flow Diagram

```
┌─────────────────┐
│  TalentMatch API│
│  (localhost:5000)│
└────────┬────────┘
         │ GET job listings/applications
         ▼
┌─────────────────┐
│  Data Sync API  │ ◄──── User clicks "Sync All Data"
│  /api/sync/all  │
└────────┬────────┘
         │ Store
         ▼
┌─────────────────┐
│    MongoDB      │
│ - job_listings  │
│ - applications  │
│ - resumes       │
└────────┬────────┘
         │ Fetch for screening
         ▼
┌─────────────────┐
│  Screening API  │ ◄──── User clicks "Run Screening"
│  /screen        │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│         ScreeningService Pipeline               │
│                                                 │
│  1. Evaluate Candidates (LLM + Fallback)       │
│     ├─ LLMService.evaluate_candidate()          │
│     ├─ Ollama (Llama3) scores 6 criteria        │
│     └─ Rule-based fallback                      │
│                                                 │
│  2. Rank by Score                               │
│     └─ Sort descending by total_weighted_score  │
│                                                 │
│  3. ⭐ Fairness Analysis (NEW)                   │
│     ├─ FairnessAgent.execute()                  │
│     ├─ Compute Disparate Impact Ratio           │
│     ├─ Run Counterfactual Tests                 │
│     └─ Generate FairnessReport                  │
│                                                 │
│  4. Store Results                               │
│     ├─ Save to screening_results                │
│     └─ Save to fairness_reports                 │
└─────────┬───────────────────────────────────────┘
          │
          ▼
┌─────────────────┐
│    MongoDB      │
│ - screening_    │
│   results       │
│ - fairness_     │ ◄──── GET /api/data/jobs/{id}/fairness
│   reports       │
└────────┬────────┘
         │ Display
         ▼
┌─────────────────┐
│   Web Portal    │
│ - Ranked Table  │
│ - Fairness Card │ ◄──── User views results
│ - Detail View   │
└─────────────────┘
```

---

## Key Agents

### 1. **Screening Agent** (`screening_service.py`)
- **Purpose**: Evaluate candidates using LLM intelligence
- **Input**: Job requirements + Candidate text
- **Output**: Scored evaluations (6 criteria)
- **LLM Model**: Llama3 via Ollama
- **Fallback**: Rule-based keyword matching

### 2. **Fairness Agent** (`fairness_agent.py`) ⭐ **NOW INTEGRATED**
- **Purpose**: Ensure unbiased rankings
- **Input**: Ranked candidates + Demographics
- **Output**: FairnessReport with compliance status
- **Metrics**:
  - Disparate Impact Ratio (DIR ≥ 0.8)
  - Counterfactual variance (0% target)
  - Demographic parity
- **Protected Attributes**: Gender, Age, Ethnicity, Nationality

### 3. **Explanation Agent** (LLM)
- **Purpose**: Generate human-readable explanations
- **Input**: Candidate scores + Job context
- **Output**: Summary, strengths, concerns
- **Model**: Llama3 via Ollama

---

## MongoDB Collections

| Collection | Purpose | Key Fields |
|------------|---------|------------|
| `job_listings` | Job postings from TalentMatch | _id, title, company, qualifications, responsibilities |
| `applications` | Student applications | student_id, job_id, cover_letter, applied_at |
| `resumes` | Parsed resume data | student_id, raw_text, parsed_data |
| `screening_results` | Candidate evaluations | job_id, candidate_id, criteria_scores, recommendation |
| **`fairness_reports`** ⭐ | Fairness analysis | job_id, is_compliant, metrics, violations |

---

## API Endpoints Summary

### Data Sync
- `POST /api/sync/all` - Sync all data from TalentMatch
- `POST /api/sync/jobs/{id}/applications` - Sync applications for specific job

### Screening
- `POST /api/screening/jobs/{id}/screen` - Run screening pipeline
- `GET /api/data/jobs/{id}/results` - Get screening results

### **Fairness** ⭐ NEW
- `GET /api/data/jobs/{id}/fairness` - Get fairness report

### Portal
- `GET /` - Dashboard
- `GET /jobs` - Job listings
- `GET /jobs/{id}` - Job detail with ranked candidates
- `GET /jobs/{id}/candidates/{candidate_id}` - Candidate detail

### Health
- `GET /api/health` - Service health
- `GET /api/llm/status` - LLM status
- `POST /api/llm/initialize` - Initialize LLM

---

## Fairness Integration - What Changed

### Before (Missing Fairness)
```
1. Sync data → MongoDB
2. User clicks "Screen"
3. Evaluate candidates (LLM)
4. Rank by score
5. ❌ NO FAIRNESS CHECK
6. Store results
7. Display rankings
```

### After (With Fairness) ⭐
```
1. Sync data → MongoDB
2. User clicks "Screen"
3. Evaluate candidates (LLM)
4. Rank by score
5. ✅ RUN FAIRNESS ANALYSIS
   - Compute DIR for gender, age, ethnicity, nationality
   - Run counterfactual tests
   - Generate compliance report
6. Store results + fairness report
7. Display rankings + fairness card
```

---

## Fairness Metrics Explained

### Disparate Impact Ratio (DIR)
- **Formula**: `P(selected | minority) / P(selected | majority)`
- **Target**: ≥ 0.8 (Four-Fifths Rule - EEOC standard)
- **Example**:
  - If 50% of Group A selected and 40% of Group B selected
  - DIR = 0.4 / 0.5 = 0.8 ✅ COMPLIANT

### Counterfactual Fairness
- **Process**:
  - Take a candidate's profile
  - Change protected attribute (e.g., gender: male → female)
  - Re-run screening
  - Verify rank doesn't change
- **Target**: 0% variance (rank should not change)

### Demographic Parity
- **Measure**: Equal selection rates across demographic groups
- **Target**: 1.0 (perfect parity)
- **Formula**: `1.0 - (max_rate - min_rate)`

---

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Screening Speed | 120 candidates in ≤30 min | ⚠️ Needs optimization |
| Historical Agreement | ≥90% | ⚠️ Validation Agent needed |
| Disparate Impact Ratio | ≥0.8 | ✅ Implemented |
| Counterfactual Variance | 0% | ✅ Implemented |
| Structured Explanations | 100% | ✅ LLM-powered |

---

## Next Steps

### Immediate
1. ✅ Fairness Agent integrated into screening workflow
2. ✅ Fairness API endpoint created
3. ✅ Web portal displays fairness metrics

### Pending
1. ⚠️ **Validation Agent**: Compare with historical hiring decisions (90% agreement target)
2. ⚠️ **Performance Optimization**: Batch processing, caching, parallel execution
3. ⚠️ **Demographic Data Collection**: Applications need gender/age/ethnicity fields
4. ⚠️ **Testing**: End-to-end integration tests with real data

---

## How to Use

### 1. Start Services
```bash
# Start MongoDB
docker run -d -p 27017:27017 --name mongodb-screening mongo:latest

# Start Ollama
ollama serve

# Pull Llama3 model
ollama pull llama3

# Start TalentMatch API (must be running on localhost:5000)
# ... start your TalentMatch service ...

# Start Flask app
python src/main.py
```

### 2. Sync Data
1. Open browser: `http://localhost:8000`
2. Click "Sync All Data" button
3. Wait for jobs and applications to sync

### 3. Run Screening
1. Click on a job in the jobs list
2. Review applications
3. Click "Run Screening" button
4. Wait for LLM evaluation (may take 1-2 min)
5. **View ranked candidates table**
6. **View fairness report card** (auto-loads)

### 4. Review Fairness
- Check compliance status (green ✅ or red ❌)
- Review Disparate Impact Ratio (should be ≥0.8)
- Read violations and recommendations if any
- Examine attribute variance table

---

## Tech Stack

- **Backend**: Flask (Python 3.10+)
- **Database**: MongoDB (with Motor async driver)
- **LLM**: Llama3 via Ollama (local inference)
- **Embeddings**: sentence-transformers
- **Frontend**: Bootstrap 5 + Jinja2 templates
- **Data Source**: TalentMatch API (REST)
- **Document Parsing**: PyPDF2, python-docx
- **Fairness**: Custom implementation (DIR, counterfactual)

---

## Configuration (.env)

```bash
# Ollama LLM
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3:latest

# TalentMatch API
TALENTMATCH_API_URL=http://localhost:5000
TALENTMATCH_API_TIMEOUT=30

# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=resume_screening

# Fairness Thresholds
FAIRNESS_DIR_THRESHOLD=0.8
FAIRNESS_VARIANCE_THRESHOLD=0.001
```

---

## Summary: Fairness in the Workflow

**Previously**: Fairness agent existed in the codebase but was **NEVER CALLED** during screening.

**Now**:
1. After candidates are evaluated and ranked
2. **FairnessAgent automatically executes**
3. Computes DIR and runs counterfactual tests
4. Generates compliance report
5. Stores report in MongoDB
6. Web portal displays fairness metrics
7. Recruiters see if hiring process is fair **before making decisions**

This ensures **every screening job** includes fairness analysis as a mandatory step, not an optional add-on.
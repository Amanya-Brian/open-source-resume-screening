# Troubleshooting: All Candidates Getting Score of 2

## Problem Description
During demo, all candidates were scored as 2 out of 5 on all criteria, indicating the LLM evaluation is failing.

---

## Root Cause Analysis

When scores are all 2/5, it means:
1. ✅ The system is NOT crashing
2. ❌ But LLM evaluation is failing
3. ✅ Fallback to default scores (all 2s) is working

**The issue**: `LLMService._default_evaluation()` returns all scores as 2 when LLM fails.

---

## Diagnostic Steps

### Step 1: Check if Ollama is Running

```bash
# Check if Ollama service is running
ollama list

# If not running, start it
ollama serve
```

**Expected**: Should show list of installed models including `llama3:latest`

### Step 2: Test Ollama Directly

```bash
# Test Ollama API
curl http://localhost:11434/api/tags

# Should return JSON with models list
```

### Step 3: Check LLM Status via API

```bash
# Get LLM status
curl http://localhost:8000/api/llm/status

# Should return:
# {
#   "status": "ready",
#   "model": "llama3:latest",
#   "ollama_url": "http://localhost:11434",
#   "available_models": ["llama3:latest", ...]
# }
```

### Step 4: Test LLM Evaluation

```bash
# Run test evaluation with sample data
curl -X POST http://localhost:8000/api/llm/test

# Should return:
# {
#   "status": "success",
#   "result": {
#     "scores": [
#       {"criterion": "education", "score": 3-5, "evidence": "..."},
#       ...
#     ],
#     "strengths": [...],
#     "concerns": [...]
#   },
#   "scores_count": 4
# }
```

**If this fails**, the issue is with Ollama or the LLM itself.

### Step 5: Check Application Logs

```bash
# Look for these log messages:
# - "Ollama is not running"
# - "LLM evaluation failed"
# - "LLM response missing scores"
# - "No LLM score found for {criterion}, defaulting to 2"

# Windows
type logs\app.log | findstr "LLM"

# Linux/Mac
tail -f logs/app.log | grep "LLM"
```

---

## Common Issues and Fixes

### Issue 1: Ollama Not Running

**Symptoms**:
- Logs show: "Ollama is not running"
- All scores are 2
- `/api/llm/status` returns "not_running"

**Fix**:
```bash
ollama serve
```

---

### Issue 2: Model Not Downloaded

**Symptoms**:
- Ollama running but model not found
- Error: "model llama3:latest not found"

**Fix**:
```bash
ollama pull llama3:latest

# Or use a smaller model for faster inference
ollama pull llama3.1:8b
```

Update `.env`:
```bash
OLLAMA_MODEL=llama3.1:8b
```

---

### Issue 3: LLM Returning Invalid JSON

**Symptoms**:
- Logs show: "Failed to parse JSON from response"
- Logs show raw LLM response with extra text

**Fix**: The LLM sometimes adds explanatory text before/after JSON. We now handle this with regex extraction, but if it persists:

1. Check prompt in [llm_service.py:148](src/services/llm_service.py#L148)
2. Ensure it says "You MUST respond with ONLY valid JSON"
3. Try a different model (some models follow instructions better)

---

### Issue 4: Criterion Name Mismatch

**Symptoms**:
- Logs show: "No LLM score found for {criterion_key}, defaulting to 2"
- LLM returns scores but they don't match our criteria

**Fix**: We now use fuzzy matching (handles underscores, case differences). If still failing:

Check LLM response format:
```python
# Expected format:
{"criterion": "education", "score": 3, "evidence": "..."}

# NOT:
{"name": "education", "value": 3}  # Wrong keys
{"criterion": "Education Level", "score": 3}  # Wrong name
```

---

### Issue 5: Ollama Timeout

**Symptoms**:
- Screening takes >2 minutes per candidate
- Eventually returns all 2s
- Logs show timeout errors

**Fix**: Increase timeout or use faster model

```python
# In llm_service.py:111
response = requests.post(
    f"{self.ollama_url}/api/chat",
    timeout=120,  # Increase to 180 or 240
)
```

Or switch to faster model:
```bash
ollama pull llama3.1:8b  # Smaller, faster
```

---

## Verification Checklist

Run these checks in order:

- [ ] 1. Ollama service is running (`ollama serve`)
- [ ] 2. Model is downloaded (`ollama list` shows llama3:latest)
- [ ] 3. Ollama API responds (`curl http://localhost:11434/api/tags`)
- [ ] 4. Flask app LLM status is "ready" (`GET /api/llm/status`)
- [ ] 5. Test evaluation works (`POST /api/llm/test`)
- [ ] 6. Application logs show "LLM evaluation successful"
- [ ] 7. Screening produces varied scores (not all 2s)

---

## Quick Fix Script

```bash
#!/bin/bash
# quick_fix_llm.sh

echo "=== Checking Ollama ==="
if ! pgrep -x "ollama" > /dev/null; then
    echo "❌ Ollama not running. Starting..."
    ollama serve &
    sleep 3
else
    echo "✅ Ollama is running"
fi

echo ""
echo "=== Checking Model ==="
if ollama list | grep -q "llama3"; then
    echo "✅ Llama3 model found"
else
    echo "❌ Llama3 not found. Downloading..."
    ollama pull llama3:latest
fi

echo ""
echo "=== Testing Ollama API ==="
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama API responding"
else
    echo "❌ Ollama API not responding"
    exit 1
fi

echo ""
echo "=== Testing LLM Evaluation ==="
curl -X POST http://localhost:8000/api/llm/test

echo ""
echo "=== Done! Try screening again ==="
```

---

## What We Fixed

1. **Added Ollama availability check** before evaluation
2. **Better logging** of raw LLM responses
3. **Fuzzy criterion matching** (handles underscores, case)
4. **Diagnostic endpoint** (`POST /api/llm/test`) for testing
5. **Detailed error messages** to pinpoint exact failure

---

## Expected Behavior After Fix

### Before (All 2s):
```json
{
  "candidate_id": "123",
  "criteria_scores": [
    {"criterion": "education", "score": 2, "evidence": "Evaluation unavailable"},
    {"criterion": "experience", "score": 2, "evidence": "Evaluation unavailable"},
    {"criterion": "technical_skills", "score": 2, "evidence": "Evaluation unavailable"},
    ...
  ],
  "total_weighted_score": 2.0,
  "percentage": 40.0,
  "recommendation": "No"
}
```

### After (Varied scores):
```json
{
  "candidate_id": "123",
  "criteria_scores": [
    {"criterion": "education", "score": 4, "evidence": "Master's degree in relevant field"},
    {"criterion": "experience", "score": 5, "evidence": "8+ years in similar role"},
    {"criterion": "technical_skills", "score": 3, "evidence": "Proficient in required tools"},
    {"criterion": "industry_knowledge", "score": 4, "evidence": "Extensive domain expertise"},
    {"criterion": "leadership", "score": 3, "evidence": "Led team of 5"},
    {"criterion": "communication", "score": 4, "evidence": "Well-structured cover letter"}
  ],
  "total_weighted_score": 3.95,
  "percentage": 79.0,
  "recommendation": "YES"
}
```

---

## Still Not Working?

If scores are still all 2s after following this guide:

1. **Check logs** for the exact error message
2. **Run diagnostic test**: `POST /api/llm/test`
3. **Try rule-based fallback**: Set `use_llm=False` in screening_service.py
4. **Contact support** with:
   - Output of `/api/llm/status`
   - Output of `/api/llm/test`
   - Last 50 lines of application logs
   - Ollama version: `ollama --version`

---

## Performance Notes

**Expected LLM evaluation time**:
- Per candidate: 5-10 seconds
- 10 candidates: 50-100 seconds (~1.5 min)
- 50 candidates: 250-500 seconds (~5-8 min)

If slower, consider:
- Using smaller model (llama3.1:8b instead of llama3:latest)
- Enabling GPU acceleration
- Batch processing (coming in next update)
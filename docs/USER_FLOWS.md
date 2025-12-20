# 🔄 User Flows & Scenarios

Complete guide to common workflows and use cases for the Dental AI Platform.

---

## 🎯 Quick Navigation

- [First-Time Setup](#-first-time-setup)
- [Daily Usage Scenarios](#-daily-usage-scenarios)
- [Tab-Specific Workflows](#-tab-specific-workflows)
- [Advanced Use Cases](#-advanced-use-cases)
- [Troubleshooting Flows](#-troubleshooting-flows)

---

## 🚀 First-Time Setup

### Scenario: Brand New Installation

**Goal:** Get the platform running for the first time

```
1. Prerequisites Check
   └─ Verify Python 3.8+ installed
      └─ Command: python3 --version
         ├─ ✅ Version 3.8+: Continue
         └─ ❌ Version < 3.8: Install Python first

2. Clone/Download Project
   └─ Navigate to: dk_project_2/backend/

3. Get API Keys
   ├─ OpenAI: https://platform.openai.com/api-keys
   ├─ Groq: https://console.groq.com/keys
   └─ Google AI: https://makersuite.google.com/app/apikey

4. Configure Environment
   └─ Edit backend/.env
      └─ Paste API keys (one per line)

5. Run Setup Script
   └─ Command: ./setup.sh
      ├─ Creates virtual environment
      ├─ Installs dependencies (~2-5 min)
      └─ Shows success message

6. Test Installation
   └─ Command: python test_example.py
      ├─ ✅ All checks pass: Ready!
      └─ ❌ Errors: See troubleshooting

7. Launch App
   ├─ Enhanced: ./run_enhanced.sh
   └─ Basic: ./run.sh

8. Open Browser
   └─ Navigate to: http://localhost:7860
```

**Time:** ~10-15 minutes
**Difficulty:** Easy

---

## 🌅 Daily Usage Scenarios

### Scenario 1: Analyze Single X-Ray (Dentist)

**User:** Dr. Sarah, general dentist
**Goal:** Check patient X-ray for wisdom teeth before referral

```
Flow:
1. Launch App
   └─ ./run_enhanced.sh

2. Navigate to Tab 1
   └─ "🔍 Wisdom Tooth Detection"

3. Upload X-Ray
   └─ Drag & drop or click upload
      └─ Select: patient_xray_2024.jpg

4. Select Model
   ├─ Quick check: Gemini Vision (faster, cheaper)
   └─ Critical case: GPT-4o Vision (more detailed)

5. Click "Analyze X-Ray"
   └─ Wait: 5-15 seconds

6. Review Results
   ├─ View annotated image (bounding boxes)
   ├─ Read analysis text
   │   ├─ Number of wisdom teeth
   │   ├─ Positions (upper/lower, left/right)
   │   └─ Descriptions (impacted, etc.)
   └─ Make clinical decision

7. Save Results (optional)
   └─ Right-click image → Save
```

**Time:** 2-3 minutes per X-ray
**Cost:** ~$0.02 (Gemini) or ~$0.10 (GPT-4o)

---

### Scenario 2: Research Multiple Models (Researcher)

**User:** Alex, dental AI researcher
**Goal:** Compare how different AI models answer the same question

```
Flow:
1. Launch App
   └─ ./run_enhanced.sh

2. Navigate to Tab 2
   └─ "💬 Multi-Model Chatbot"

3. Formulate Research Question
   └─ Example: "What are the indications for wisdom tooth extraction?"

4. Enter Question
   └─ Type in text box

5. Click "Ask All Models"
   └─ Wait: 5-8 seconds (parallel execution)

6. Compare Responses
   ├─ OpenAI GPT-4o (left column)
   ├─ Google Gemini (middle column)
   └─ Groq Llama3 (right column)

7. Analyze Differences
   ├─ Note unique perspectives
   ├─ Compare accuracy
   ├─ Evaluate response styles
   └─ Document findings

8. Try More Questions
   └─ Or use example questions
```

**Time:** 30 seconds per question
**Cost:** ~$0.01 per query (all 3 models)

---

### Scenario 3: Dataset Exploration (Student)

**User:** Jamie, dental student
**Goal:** Practice X-ray interpretation using dataset

```
Flow:
1. Launch Enhanced App
   └─ ./run_enhanced.sh

2. Navigate to Tab 3
   └─ "📊 Dataset Explorer"

3. Load Dataset (first time only)
   └─ Click: "📥 Load Dataset from Hugging Face"
      └─ Wait: 10-30 seconds (downloads 90 MB)
      └─ Cache: Instant on subsequent loads

4. View Dataset Stats
   ├─ Total: 1,206 samples
   ├─ Label distribution
   └─ Image specifications

5. Browse Samples
   ├─ Option A: Sequential
   │   └─ Click: "Next ➡️" / "⬅️ Previous"
   ├─ Option B: Random
   │   └─ Click: "🎲 Random Sample"
   └─ Option C: Jump to specific
       └─ Enter index → "Go to Index"

6. Practice Interpretation
   ├─ View X-ray
   ├─ Make own assessment
   ├─ Note findings
   └─ Move to next sample

7. Self-Test with AI
   └─ Run batch analysis to check answers
```

**Time:** 5-10 minutes per session
**Cost:** Free (browsing only)

---

## 📊 Tab-Specific Workflows

### Tab 1: Wisdom Tooth Detection Workflows

#### Workflow 1A: Quick Screening
```
User: Busy clinic needing fast results
Model: Gemini Vision
Process:
  1. Upload X-ray
  2. Auto-select Gemini (default)
  3. Click analyze
  4. Get results in 5-8 seconds
  5. Move to next patient
```

#### Workflow 1B: Detailed Analysis
```
User: Complex case needing thorough review
Model: GPT-4o Vision
Process:
  1. Upload X-ray
  2. Select GPT-4o Vision
  3. Click analyze
  4. Wait 10-15 seconds
  5. Review detailed descriptions
  6. Use for treatment planning
```

#### Workflow 1C: Second Opinion
```
User: Uncertain about initial findings
Process:
  1. Analyze with Gemini Vision
  2. Note results
  3. Re-analyze same image with GPT-4o
  4. Compare findings
  5. Look for consensus or differences
```

---

### Tab 2: Multi-Model Chatbot Workflows

#### Workflow 2A: Medical Question
```
User: Clinician with patient question
Process:
  1. Enter patient's question
  2. Ask all 3 models
  3. Synthesize consensus answer
  4. Identify any outliers
  5. Use clinically appropriate response
```

#### Workflow 2B: Research Query
```
User: Researcher comparing AI capabilities
Process:
  1. Prepare standardized questions
  2. Ask each question to all models
  3. Record response times
  4. Compare accuracy/completeness
  5. Document model strengths/weaknesses
```

#### Workflow 2C: Educational Use
```
User: Instructor teaching dental AI
Process:
  1. Demonstrate to students
  2. Show same question → 3 different answers
  3. Discuss AI variability
  4. Analyze response quality
  5. Teach critical evaluation skills
```

---

### Tab 3: Dataset Explorer Workflows

#### Workflow 3A: Dataset Familiarization
```
User: New user exploring dataset
Process:
  1. Load dataset
  2. Review statistics
  3. Browse 10-20 samples
  4. Note image quality/variety
  5. Understand label distribution
```

#### Workflow 3B: Batch Processing for Research
```
User: Researcher analyzing patterns
Process:
  1. Load dataset
  2. Set batch parameters:
     - Start: 0
     - Size: 50
     - Model: Gemini Vision
  3. Run batch analysis
  4. Wait ~3-5 minutes
  5. Review aggregate results
  6. Export to JSON
  7. Analyze in Python/Excel
```

#### Workflow 3C: Model Comparison Study
```
User: Comparing GPT-4o vs Gemini on same data
Process:
  1. Batch 1: Samples 0-100, GPT-4o Vision
     └─ Note: Time, cost, accuracy
  2. Batch 2: Samples 0-100, Gemini Vision
     └─ Note: Time, cost, accuracy
  3. Compare results:
     ├─ Detection rates
     ├─ False positives/negatives
     ├─ Processing time
     └─ Cost per sample
  4. Statistical analysis
```

---

## 🎓 Advanced Use Cases

### Use Case 1: Clinical Practice Integration

**Scenario:** Dental clinic integrates platform into workflow

```
Setup:
1. Install on clinic workstation
2. Configure with clinic API keys
3. Create standard operating procedure (SOP)

Daily Workflow:
  Morning:
  └─ Launch app: ./run_enhanced.sh

  For Each Patient:
  ├─ Export X-ray from PACS system
  ├─ Upload to Tab 1
  ├─ Run analysis (Gemini for speed)
  ├─ Screenshot results
  └─ Add to patient record

  End of Day:
  └─ Close app

SOP:
  ✓ Use Gemini Vision for routine screening
  ✓ Use GPT-4o for complex cases
  ✓ Always verify AI findings clinically
  ✓ Document AI use in patient notes
```

---

### Use Case 2: Research Study on AI Accuracy

**Scenario:** Validate AI detection accuracy vs. expert radiologists

```
Study Design:
  Sample Size: 200 X-rays from dataset
  Gold Standard: Board-certified oral radiologist
  AI Models: GPT-4o Vision vs Gemini Vision

Workflow:
  Phase 1: Expert Annotation (Manual)
  ├─ Radiologist reviews all 200 samples
  ├─ Marks wisdom teeth locations
  └─ Records ground truth

  Phase 2: AI Batch Processing
  ├─ Batch 1: Samples 0-200, GPT-4o
  │   └─ Export results → gpt4_results.json
  ├─ Batch 2: Samples 0-200, Gemini
  │   └─ Export results → gemini_results.json

  Phase 3: Statistical Analysis
  ├─ Calculate sensitivity/specificity
  ├─ Measure inter-rater agreement (Kappa)
  ├─ Compare detection rates
  └─ Publish findings

Tools Used:
  - Tab 3: Batch processing
  - Python: Statistical analysis
  - Excel: Data visualization
```

---

### Use Case 3: Educational Course Development

**Scenario:** Create dental AI interpretation course

```
Course Structure:
  Module 1: Introduction to Dental AI
  └─ Use Tab 2 to demonstrate AI capabilities

  Module 2: X-Ray Interpretation Basics
  └─ Use Tab 3 to browse dataset examples

  Module 3: Hands-On Practice
  ├─ Students upload X-rays to Tab 1
  ├─ Make their own assessment first
  ├─ Then compare with AI results
  └─ Discuss discrepancies

  Module 4: AI Model Comparison
  └─ Use Tab 2 to compare model responses

  Module 5: Batch Analysis Project
  ├─ Students select 20 samples
  ├─ Run batch analysis
  ├─ Write report on findings
  └─ Present to class

Assessment:
  - X-ray interpretation accuracy
  - AI result interpretation
  - Critical evaluation skills
  - Research project quality
```

---

## 🔧 Troubleshooting Flows

### Issue: App Won't Start

```
1. Check Python Version
   └─ python3 --version
      ├─ < 3.8: Upgrade Python
      └─ ≥ 3.8: Continue

2. Check Virtual Environment
   └─ ls backend/venv/
      ├─ Missing: Run ./setup.sh
      └─ Exists: Continue

3. Activate Environment
   └─ source venv/bin/activate
      └─ (venv) should appear in prompt

4. Check Dependencies
   └─ pip list | grep gradio
      ├─ Missing: pip install -r requirements.txt
      └─ Installed: Continue

5. Check Port
   └─ lsof -i:7860
      ├─ In use: Kill process or change port
      └─ Free: Continue

6. Try Running Again
   └─ python dental_ai_enhanced.py
      └─ Check error message
```

---

### Issue: API Errors

```
1. Check API Keys
   └─ cat backend/.env
      ├─ Empty/wrong: Add correct keys
      └─ Looks good: Continue

2. Test Individual APIs
   └─ python test_example.py
      └─ Note which APIs fail

3. Verify Key Status
   For failing APIs:
   ├─ OpenAI: Check https://platform.openai.com/account/api-keys
   ├─ Groq: Check https://console.groq.com/keys
   └─ Google: Check https://console.cloud.google.com/apis/credentials

4. Check Rate Limits
   └─ Review API dashboard
      ├─ Exceeded: Wait or upgrade
      └─ OK: Continue

5. Check Internet
   └─ ping api.openai.com
      ├─ No connection: Fix network
      └─ Connected: Contact API support
```

---

### Issue: Dataset Won't Load

```
1. Check Internet Connection
   └─ ping huggingface.co
      ├─ Failed: Fix connection
      └─ Success: Continue

2. Clear Cache
   └─ rm -rf ~/.cache/huggingface/datasets/
      └─ Try loading again

3. Check Disk Space
   └─ df -h
      ├─ < 1 GB free: Free up space
      └─ OK: Continue

4. Manual Download Test
   └─ Run in Python:
      from datasets import load_dataset
      ds = load_dataset("RayanAi/Main_teeth_dataset")
      └─ Note specific error

5. Check HuggingFace Status
   └─ Visit: https://status.huggingface.co/
      └─ If down: Wait and retry
```

---

## ⏱️ Time & Cost Estimates

### Per-Operation Times

| Operation | Time | Notes |
|-----------|------|-------|
| Single X-ray (Gemini) | 5-8s | Fastest |
| Single X-ray (GPT-4o) | 10-15s | Most detailed |
| Multi-chat query | 5-8s | Parallel execution |
| Load dataset (first time) | 10-30s | Downloads 90 MB |
| Load dataset (cached) | <1s | Instant |
| Batch 10 samples (Gemini) | ~60s | 6s per sample |
| Batch 10 samples (GPT-4o) | ~120s | 12s per sample |

### Cost Estimates

| Operation | GPT-4o | Gemini | Groq |
|-----------|--------|--------|------|
| Single X-ray | ~$0.10 | ~$0.02 | N/A |
| Chat query | ~$0.005 | ~$0.001 | FREE |
| Batch 100 X-rays | ~$10 | ~$2 | N/A |
| Batch 1000 X-rays | ~$100 | ~$20 | N/A |

**Recommendation:** Use Gemini for exploration, GPT-4o for critical cases

---

## 📱 Mobile/Remote Access

### Local Network Access

```
1. Find IP Address
   └─ ip addr show | grep inet
      └─ Note: 192.168.1.X

2. Launch with Network Access
   └─ Edit dental_ai_enhanced.py:
      server_name="0.0.0.0"  # Already set ✓

3. Access from Other Device
   └─ Navigate to: http://192.168.1.X:7860
      └─ Must be on same network
```

### Cloud Deployment (Advanced)

```
Options:
  A. Gradio Share Link
     └─ Set: share=True in demo.launch()
        ├─ Pros: Instant, no setup
        └─ Cons: Temporary, public URL

  B. Hugging Face Spaces
     └─ Deploy to: https://huggingface.co/spaces
        ├─ Pros: Free hosting, persistent
        └─ Cons: Requires HF account

  C. Cloud VM (AWS/GCP/Azure)
     └─ Deploy to VM with public IP
        ├─ Pros: Full control, production-ready
        └─ Cons: Cost, setup complexity
```

---

**User Flows Documentation Complete!**
*Use this guide to navigate common scenarios and maximize platform value.*

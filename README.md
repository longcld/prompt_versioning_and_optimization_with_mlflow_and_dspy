# 🚀 LLMOps: Prompt Versioning and Optimization with MLflow and DSPy

A comprehensive MLOps framework for prompt engineering, versioning, and automated optimization using MLflow prompt registry and DSPy optimization techniques. This project demonstrates production-ready LLM pipeline management with business scope classification as a use case.

## 🎯 Overview

This project showcases enterprise-grade LLMOps practices by implementing a **Business Filter Classification System** that determines whether user questions fall within predefined business scopes. It demonstrates:

- **Prompt Management**: Version control and registry using MLflow
- **Automated Optimization**: DSPy-powered prompt optimization with LLM-as-judge evaluation
- **Robust Evaluation**: Semantic evaluation metrics with feedback loops
- **Production Pipeline**: End-to-end ML pipeline with experiment tracking

## 🏗️ Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Data Ingestion    │    │  Prompt Registry    │    │   Model Pipeline    │
│                     │    │                     │    │                     │
│ • BOT_FAQS.json     │───▶│ • MLflow Registry   │───▶│ • DSPy Signatures   │
│ • business_filter   │    │ • Version Control   │    │ • GPT-4o Integration│
│   .xlsx             │    │ • Template Storage  │    │ • Business Logic    │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                      │                           │
                                      ▼                           ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Optimization      │    │    Evaluation       │    │    Results &        │
│                     │    │                     │    │   Visualization     │
│ • GEPA Optimizer    │    │ • LLM-as-Judge      │    │ • Performance       │
│ • Teacher Models    │    │ • Feedback Loops    │    │   Analytics         │
│ • Auto-tuning       │    │ • Semantic Metrics  │    │ • Comparison Charts │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## 🔧 Key Features

### 🎨 Prompt Engineering & Versioning
- **MLflow Prompt Registry**: Centralized prompt storage with version control
- **Template Management**: Structured prompt templates with dynamic content
- **A/B Testing**: Compare different prompt versions systematically

### 🤖 Automated Optimization
- **DSPy Integration**: Automated prompt optimization using GEPA (Generate, Evaluate, Prompt, Adapt)
- **Multi-Model Support**: GPT-4o, GPT-4.1, and GPT-5 integration
- **Teacher-Student Architecture**: Advanced model distillation for optimization

### 📊 Comprehensive Evaluation
- **LLM-as-Judge**: Semantic evaluation using advanced language models
- **Feedback Systems**: Automated feedback generation for continuous improvement
- **Performance Tracking**: Detailed metrics and comparison visualizations

### 🏢 Business Logic Implementation
- **Scope Classification**: Intelligent business boundary detection
- **Multi-Category Support**: Handle different business domains (Cards, Payment Accounts, etc.)
- **Vietnamese Language Support**: Specialized handling for Vietnamese text processing

## 📦 Installation

### Prerequisites
- Python 3.8+
- OpenAI API key
- MLflow server (optional, defaults to local)

### Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd prompt_versioning_and_optimization_with_mlflow

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Environment Configuration

Create a `.env` file with the following variables:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (defaults provided)
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

## 🚀 Quick Start

### 1. Initialize Prompt Registry

Register your initial prompts with MLflow:

```bash
python init_prompt.py
```

This will:
- Register base business filter prompts
- Create versioned prompt templates
- Set up MLflow tracking

### 2. Run Baseline Evaluation

Execute the optimization pipeline:

```bash
python optimization.py
```

This will:
- Load the latest prompt from MLflow registry
- Create DSPy pipeline with business logic
- Run baseline evaluation with sample data
- Generate performance metrics

## 📊 Usage Examples

### Basic Business Filter Classification

```python
from optimization import BusinessFilter

# Initialize the filter
business_filter = BusinessFilter()

# Classify a question
question = "Tôi muốn đổi mật khẩu iPay"
result = business_filter(question=question)

print(f"In Scope: {result.in_scope}")
print(f"Reason: {result.reason}")
```

### Prompt Optimization

```python
from dspy import GEPA
from evaluations.business_filter import metric_with_feedback

# Set up optimizer
optimizer = GEPA(
    metric=metric_with_feedback,
    auto="light",
    num_threads=32,
    track_stats=True
)

# Optimize the prompt
optimized_program = optimizer.compile(
    business_filter,
    trainset=train_set,
    valset=eval_set
)
```

### MLflow Integration

```python
from mlflow.genai import register_prompt, load_prompt

# Register a new prompt version
prompt_version = register_prompt(
    name="business_filter_v2",
    template=new_template,
    commit_message="Improved accuracy for payment queries"
)

# Load latest prompt
latest_prompt = load_prompt("prompts:/business_filter@latest")
```

## 📁 Project Structure

```
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Configuration and setup
├── 📄 init_prompt.py               # Prompt registry initialization
├── 📄 optimization.py              # Main optimization pipeline
├── 📄 example.ipynb                # Interactive notebook
│
├── 📁 data/                        # Dataset storage
│   ├── BOT_FAQS.json              # FAQ dataset
│   └── business_filter.xlsx        # Business scope definitions
│
├── 📁 dataloaders/                 # Data processing modules
│   └── business_filter.py          # Business data loader
│
├── 📁 evaluations/                 # Evaluation metrics
│   ├── business_filter.py          # Classification metrics
│   └── llm_as_judge.py            # LLM evaluation logic
│
└── 📁 results/                     # Output and analytics
    ├── absa_llm_judge_results/     # Evaluation results
    ├── business_filter_card_results/
    ├── business_filter_ipay_results/
    └── business_filter_payment_account_results/
```

## 🔧 Configuration

The system is highly configurable through `setup.py`. Key configuration options:

```python
CONFIG = {
    # API Configuration
    'OPENAI_API_KEY': 'your-api-key',
    'MLFLOW_URI': 'http://127.0.0.1:5000',
    
    # Model Configuration
    'MODEL_CONFIG': {
        'model': 'openai/gpt-4o-mini',
        'temperature': 0,
        'max_tokens': 16000
    },
    
    # Optimization Settings
    'EXPERIMENT_NAME': 'Business_Filter_Optimization',
    'EVAL_SAMPLE_RATE': 0.1,
    'USE_CACHING': True
}
```

## 📈 Evaluation Metrics

The system uses multiple evaluation approaches:

### 1. **Accuracy Metrics**
- Binary classification accuracy
- Precision, Recall, F1-score
- Category-specific performance

### 2. **LLM-as-Judge Evaluation**
- Semantic similarity assessment
- Reasoning quality evaluation
- Contextual appropriateness scoring

### 3. **Feedback-Driven Optimization**
- Automated feedback generation
- Error analysis and correction suggestions
- Continuous improvement loops

## 🎯 Use Cases

### 1. **Customer Service Automation**
- Classify customer inquiries by business scope
- Route questions to appropriate departments
- Filter out-of-scope requests automatically

### 2. **Content Moderation**
- Determine content relevance to business domains
- Automated content categorization
- Compliance checking for business policies

### 3. **Prompt Engineering**
- Systematic prompt optimization
- Version control for prompt templates
- A/B testing for different approaches

## 🚀 Advanced Features

### Multi-Model Architecture
```python
# Different models for different tasks
student_model = "gpt-4o-mini"      # Fast inference
teacher_model = "gpt-5-mini"       # Optimization guidance  
judge_model = "gpt-4.1-mini"       # Evaluation
```

### Automated Optimization Pipeline
```python
# GEPA (Generate, Evaluate, Prompt, Adapt) optimization
optimizer = GEPA(
    metric=metric_with_feedback,
    auto="light",                   # Optimization intensity
    num_threads=32,                 # Parallel processing
    track_stats=True,               # Performance tracking
    reflection_minibatch_size=3     # Batch optimization
)
```

### Caching and Performance
- Intelligent API call caching
- Batch processing for evaluations
- Memory-efficient data loading
- Progress tracking and logging


## 🙏 Acknowledgments

- **MLflow Team** for the excellent prompt registry and tracking capabilities
- **DSPy Community** for the powerful optimization framework
- **OpenAI** for providing state-of-the-art language models
- **Contributors** who help improve this project

## 📚 Documentation & Resources

### Related Documentation
- [MLflow Prompt Engineering Guide](https://mlflow.org/docs/latest/llms/prompt-engineering/index.html)
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

### Research Papers
- "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines"
- "Constitutional AI: Harmlessness from AI Feedback"
- "Training language models to follow instructions with human feedback"

*Making LLM prompt engineering systematic, scalable, and production-ready.*
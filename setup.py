# =============================================================================
# 🏗️ ENVIRONMENT SETUP AND CONFIGURATION
# =============================================================================


import os
import sys
from pathlib import Path

# Initialize MLflow
import mlflow
import mlflow.exceptions

print("🚀 MLflow ABSA Pipeline with LLM-as-Judge Evaluation")
print("=" * 60)
print("Using semantic evaluation for more accurate scoring")
print("=" * 60)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env")
except ImportError:
    print("ℹ️ Using system environment variables")

# Configuration
CONFIG = {
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    'MLFLOW_URI': os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'),
    'EXPERIMENT_NAME': "Business_Filter_Optimization",
    'CREATE_VISUALIZATIONS': True,
    'USE_CACHING': True  # Cache judge evaluations to save API calls
}

CONFIG["DATA_PATH"] = "data/BOT_FAQS.json"
CONFIG["CATEGORY"] = "PaymentAccount"
CONFIG["PROMPT_NAME"] = 'business_filter_payment_account'
CONFIG["OUTPUT_DIR"] = Path(f'{CONFIG["PROMPT_NAME"]}_results')
CONFIG["EVAL_SAMPLE_RATE"] = 0.1  # 10% for evaluation


CONFIG["MODEL_CONFIG"] = {
    "model": "openai/gpt-4o-mini",
    "temperature": 0,
    "max_tokens": 16000
}

CONFIG["TEACHER_MODEL_CONFIG"] = {
    "model": "openai/gpt-5-mini",
    "temperature": 1,
    "max_tokens": 64000
}

CONFIG["JUDGE_MODEL_CONFIG"] = {
    "model": "openai/gpt-4.1-mini",
    "temperature": 0,
    "max_tokens": 32000
}


def setup_and_configure():
    print("🔧 Setting up environment...")

    # Create output directory
    CONFIG['OUTPUT_DIR'].mkdir(exist_ok=True)

    # Validate configuration
    if not CONFIG['OPENAI_API_KEY']:
        print("❌ OPENAI_API_KEY not found in environment")
        print("   Please set your OpenAI API key in .env file or environment")
        sys.exit(1)

    print(f"✅ Configuration validated")
    print(f"   Model: {CONFIG['MODEL_CONFIG']['model']}")
    print(f"   Teacher: {CONFIG['TEACHER_MODEL_CONFIG']['model']}")
    print(f"   Judge: {CONFIG['JUDGE_MODEL_CONFIG']['model']}")
    print(f"   MLflow: {CONFIG['MLFLOW_URI']}")
    print(f"   Output: {CONFIG['OUTPUT_DIR']}")

    mlflow.dspy.autolog(
        log_compiles=True,    # Track optimization process
        log_evals=True,       # Track evaluation results
        log_traces_from_compile=True  # Track program traces during optimization
    )
    mlflow.set_tracking_uri(CONFIG['MLFLOW_URI'])

    mlflow.autolog()

    # Test MLflow connection
    try:
        experiments = mlflow.search_experiments()
        print(f"✅ MLflow connected: {len(experiments)} experiments found")
    except Exception as e:
        print(f"❌ MLflow connection failed: {e}")
        print("   Please ensure MLflow server is running:")
        print("   mlflow server --host 127.0.0.1 --port 5000")
        sys.exit(1)

    # Initialize OpenAI
    from openai import OpenAI
    client = OpenAI(api_key=CONFIG['OPENAI_API_KEY'])
    print("✅ OpenAI client initialized")

    # Check DSPy availability
    try:
        import dspy
        DSPY_AVAILABLE = True
        print(f"✅ DSPy available: {dspy.__version__}")
    except ImportError:
        DSPY_AVAILABLE = False
        print("⚠️ DSPy not available (pip install dspy-ai)")

    # Set up experiment
    try:
        experiment = mlflow.create_experiment(CONFIG['EXPERIMENT_NAME'])
        print(f"✅ Created experiment: {CONFIG['EXPERIMENT_NAME']}")
    except mlflow.exceptions.MlflowException:
        print(f"ℹ️ Using existing experiment: {CONFIG['EXPERIMENT_NAME']}")

    mlflow.set_experiment(CONFIG['EXPERIMENT_NAME'])
    current_experiment = mlflow.get_experiment_by_name(
        CONFIG['EXPERIMENT_NAME'])

    print(f"🧪 Active experiment: {current_experiment.name}")
    print(
        f"🌐 MLflow UI: {CONFIG['MLFLOW_URI']}/#/experiments/{current_experiment.experiment_id}"
    )
    print("✅ Environment setup complete")
    print("=" * 60)
    print()

    return CONFIG


CONFIG = setup_and_configure()

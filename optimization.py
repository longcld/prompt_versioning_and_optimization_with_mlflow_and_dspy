from dspy import GEPA
from evaluations.business_filter import evaluation_metric, metric_with_feedback
import dspy
from setup import CONFIG
from dataloaders.business_filter import train_set, eval_set, inscope, outscope
import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Tuple, Optional
import time
from datetime import datetime
import re
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from mlflow.genai import register_prompt, load_prompt, optimize_prompt
from mlflow.genai.optimize import OptimizerConfig, LLMParams
from mlflow.genai.scorers import scorer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

print("📊 Setting up the business filter prediction pipeline...")


print("📥 Loading the latest prompt from MLflow...")
prompt_from_mlflow = load_prompt((f"prompts:/{CONFIG['PROMPT_NAME']}@latest"))
print(
    f"✅ Loaded prompt version {prompt_from_mlflow.version}"
)
print("=" * 60)
print(prompt_from_mlflow.template)
print("=" * 60)
print()


class BusinessFilterSignature(dspy.Signature):
    """Determine if a question is within the business scope."""
    outscope: str = dspy.InputField()
    inscope: str = dspy.InputField()
    question: str = dspy.InputField()
    reason: str = dspy.OutputField(
        description="A concise explanation in Vietnamese, clearly referencing specific items from the <Inscope> or <Outscope> sections, detail error matched if existed. Avoid assumptions."
    )
    in_scope: bool = dspy.OutputField(
        description="Whether the question is in scope."
    )


class BusinessFilter(dspy.Module):
    def __init__(self):
        self.lm = dspy.LM(**CONFIG["MODEL_CONFIG"])
        self.teacher_lm = dspy.LM(**CONFIG["TEACHER_MODEL_CONFIG"])
        dspy.configure(lm=self.lm)

        self.inscope = inscope
        self.outscope = outscope

        self.signature = BusinessFilterSignature.with_instructions(
            prompt_from_mlflow.template)
        self.filter = dspy.Predict(self.signature)

    def forward(self, question: str) -> Dict[str, Any]:
        return self.filter(
            inscope=self.inscope,
            outscope=self.outscope,
            question=question
        )


business_filter = BusinessFilter()
print("✅ Business filter pipeline is ready")
print()
print("🤖 Running a sample prediction...")

prediction = business_filter(
    question="Chín tám Chị ờ có gọi lên tổng đài để nhờ đổi cái mật khẩu bên em ấy Mật khẩu Ipay thì ờ đổi như thế nào nhở"
)

print(prediction)
print("=" * 60)
print()


print("🏆 Run baseline evaluation...")
evaluate = dspy.Evaluate(
    devset=eval_set,
    metric=evaluation_metric,
    lm=dspy.LM(**CONFIG["JUDGE_MODEL_CONFIG"]),
    display_progress=True,
    # display_table=5
)

results = evaluate(business_filter)

# print("=" * 60)
# print()

# =============================================================================
# OPTIMIZATION
# =============================================================================


# print("🚀 Starting prompt optimization...")


# teacher_lm = dspy.LM(**CONFIG["TEACHER_MODEL_CONFIG"])

# optimizer = GEPA(
#     metric=metric_with_feedback,
#     auto="light",
#     num_threads=32,
#     track_stats=True,
#     reflection_minibatch_size=3,
#     reflection_lm=teacher_lm
# )

# optimized_program = optimizer.compile(
#     business_filter,
#     trainset=train_set,
#     valset=eval_set,
# )

# optimized_program.save(f"{CONFIG['OUTPUT_DIR'] / CONFIG['PROMPT_NAME']}.json")
# print("✅ Optimization completed")
# print()

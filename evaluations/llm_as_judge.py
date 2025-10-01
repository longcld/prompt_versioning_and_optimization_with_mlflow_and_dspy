# =============================================================================
# 🎯 LLM-AS-JUDGE EVALUATION SYSTEM
# =============================================================================

from setup import CONFIG
import os
import sys
import json
from typing import List, Dict, Any
from dataclasses import dataclass

print("🎯 Setting up LLM-as-Judge evaluation system...")

@dataclass
class EvaluationResult:
    """Structured evaluation result from LLM judge."""
    aspect_matches: List[Dict[str, Any]]
    aspect_precision: float
    aspect_recall: float
    aspect_f1: float
    sentiment_accuracy: float
    overall_score: float
    explanation: str
    num_predicted: int
    num_expected: int

class LLMJudge:
    """LLM-based semantic evaluation for ABSA."""
    
    def __init__(self, model: str = None, use_cache: bool = True):
        self.model = model or CONFIG['JUDGE_MODEL']
        self.use_cache = use_cache
        self.cache = {}
        self.api_calls = 0
        self.cache_hits = 0
        
    def evaluate(self, predicted: str, expected: str, review: str) -> EvaluationResult:
        """Evaluate predicted aspects against expected using semantic understanding."""
        
        # Check cache
        cache_key = f"{predicted}|{expected}|{review}"
        if self.use_cache and cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        # Create evaluation prompt
        eval_prompt = self._create_evaluation_prompt(predicted, expected, review)
        
        try:
            # Call LLM judge
            self.api_calls += 1
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.0,
                max_tokens=1000
            )
            
            # Parse response
            result = self._parse_judge_response(response.choices[0].message.content)
            
            # Cache result
            if self.use_cache:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            print(f"⚠️ Judge evaluation error: {e}")
            # Return zero scores on error
            return EvaluationResult(
                aspect_matches=[],
                aspect_precision=0.0,
                aspect_recall=0.0,
                aspect_f1=0.0,
                sentiment_accuracy=0.0,
                overall_score=0.0,
                explanation="Evaluation failed",
                num_predicted=0,
                num_expected=0
            )
    
    def _create_evaluation_prompt(self, predicted: str, expected: str, review: str) -> str:
        """Create prompt for LLM judge."""
        return f"""You are an expert evaluator for Aspect-Based Sentiment Analysis (ABSA).

Your task is to evaluate how well the predicted aspects match the expected aspects, considering semantic similarity rather than exact matches.

ORIGINAL REVIEW:
{review}

EXPECTED ASPECTS:
{expected}

PREDICTED ASPECTS:
{predicted}

EVALUATION GUIDELINES:
1. Consider semantic similarity - "delivery speed" and "food delivery" referring to delivery time should match
2. "staff" and "staff members" or "employees" should match
3. "price" and "pricing" or "cost" should match
4. Sentiment polarity (positive/negative/neutral) must match for credit
5. Be fair but accurate in your assessment

Please evaluate and return ONLY a JSON object with this exact structure:
{{
    "aspect_matches": [
        {{
            "expected_aspect": "aspect name from expected",
            "predicted_aspect": "matching aspect from predicted or null",
            "is_match": true/false,
            "match_reason": "why they match or don't match",
            "sentiment_match": true/false
        }}
    ],
    "metrics": {{
        "true_positives": number of correctly identified aspects with correct sentiment,
        "false_positives": number of predicted aspects that don't match any expected,
        "false_negatives": number of expected aspects not found in predicted,
        "sentiment_correct": number of aspects with correct sentiment
    }},
    "explanation": "Brief explanation of the evaluation"
}}

Return ONLY the JSON object, no other text."""

    def _parse_judge_response(self, response: str) -> EvaluationResult:
        """Parse the judge's response into structured result."""
        try:
            # Extract JSON from response
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            
            data = json.loads(response.strip())
            
            # Extract metrics
            metrics = data.get('metrics', {})
            tp = metrics.get('true_positives', 0)
            fp = metrics.get('false_positives', 0)
            fn = metrics.get('false_negatives', 0)
            
            # Calculate precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Calculate sentiment accuracy
            total_expected = tp + fn
            sentiment_correct = metrics.get('sentiment_correct', 0)
            sentiment_accuracy = sentiment_correct / total_expected if total_expected > 0 else 0.0
            
            # Calculate overall score (weighted combination)
            overall_score = 0.6 * f1 + 0.4 * sentiment_accuracy
            
            return EvaluationResult(
                aspect_matches=data.get('aspect_matches', []),
                aspect_precision=precision,
                aspect_recall=recall,
                aspect_f1=f1,
                sentiment_accuracy=sentiment_accuracy,
                overall_score=overall_score,
                explanation=data.get('explanation', ''),
                num_predicted=tp + fp,
                num_expected=tp + fn
            )
            
        except Exception as e:
            print(f"⚠️ Error parsing judge response: {e}")
            return EvaluationResult(
                aspect_matches=[],
                aspect_precision=0.0,
                aspect_recall=0.0,
                aspect_f1=0.0,
                sentiment_accuracy=0.0,
                overall_score=0.0,
                explanation=f"Parse error: {str(e)}",
                num_predicted=0,
                num_expected=0
            )
    
    def print_stats(self):
        """Print usage statistics."""
        print(f"\n📊 LLM Judge Statistics:")
        print(f"   API calls: {self.api_calls}")
        print(f"   Cache hits: {self.cache_hits}")
        print(f"   Cache rate: {self.cache_hits / (self.api_calls + self.cache_hits) * 100:.1f}%")

# Initialize global judge
judge = LLMJudge(use_cache=CONFIG['USE_CACHING'])
print("✅ LLM Judge initialized")
print()
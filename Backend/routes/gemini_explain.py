from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import json

router = APIRouter()

def get_client():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Client init error: {e}")
        return None

class ExplainRequest(BaseModel):
    bias_results: dict
    dataset_name: str = "the dataset"

class MitigationRequest(BaseModel):
    sensitive_col: str
    fairness_score: int
    demographic_parity: dict
    equalized_odds: dict

def parse_gemini_json(text: str):
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)

@router.post("/bias-summary")
async def explain_bias(req: ExplainRequest):
    client = get_client()
    bias_data = req.bias_results.get("bias_analysis", {})
    bias_text = json.dumps(bias_data, indent=2)
    
    # Detect worst-performing attributes
    worst_attrs = sorted(bias_data.items(), key=lambda x: x[1].get("fairness_score", 100))[:3]

    prompt = (
        "You are an expert AI ethics consultant analyzing fairness metrics from a machine learning model. "
        "Provide an executive summary of the bias detection results. "
        "Be specific, quantitative, and actionable. Assume the audience includes both technical and non-technical stakeholders. "
        "\n\n"
        f"Dataset: {req.dataset_name}\n"
        f"Model Accuracy: {req.bias_results.get('model_accuracy', 'N/A')}\n"
        f"Total Samples: {req.bias_results.get('total_samples', 'N/A')}\n"
        "\n"
        "BIAS ANALYSIS RESULTS:\n"
        f"{bias_text}\n"
        "\n"
        f"Worst-performing attributes (lowest fairness scores): {[attr for attr, _ in worst_attrs]}\n"
        "\n"
        "Return ONLY a valid JSON object (no markdown, no explanation outside JSON). "
        "Required fields:\n"
        "- executive_summary (2-3 sentences; explain what the metrics mean and the overall risk level)\n"
        "- attribute_explanations (object; for each worst attribute, explain disparity and real-world impact)\n"
        "- real_world_impact (1-2 sentences; concrete harms this bias could cause)\n"
        "- urgency_level (Critical if any score <40, High if <60, Medium if <80, else Low)\n"
        "- key_finding (1 sentence; the most critical finding that judges should know)"
    )

    if client:
        try:
            response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
            parsed = parse_gemini_json(response.text)
            return JSONResponse(content={"source": "gemini", "explanation": parsed})
        except Exception as e:
            print(f"Gemini bias-summary error: {e}")

    # Fallback
    overall_scores = [v.get("fairness_score", 100) for v in bias_data.values()]
    avg = sum(overall_scores) / len(overall_scores) if overall_scores else 100
    urgency = "Critical" if avg < 40 else "High" if avg < 60 else "Medium" if avg < 80 else "Low"
    worst = min(bias_data, key=lambda x: bias_data[x].get("fairness_score", 100)) if bias_data else "N/A"
    return JSONResponse(content={"source": "fallback", "explanation": {
        "executive_summary": f"Analysis of {req.dataset_name} reveals bias across {len(bias_data)} attribute(s) with average fairness score {round(avg)}/100.",
        "attribute_explanations": {k: f"{v.get('severity','Unknown')} bias. Score: {v.get('fairness_score',0)}/100." for k, v in bias_data.items()},
        "real_world_impact": "Biased models cause unfair treatment in hiring, lending, or healthcare.",
        "urgency_level": urgency,
        "key_finding": f"Most critical: '{worst}' with fairness score {bias_data.get(worst, {}).get('fairness_score', '?')}/100."
    }})


@router.post("/mitigation")
async def suggest_mitigation(req: MitigationRequest):
    client = get_client()
    col = req.sensitive_col
    
    # Context for better recommendations
    dp_disparity = req.demographic_parity.get("disparity", 0)
    eo_tpr_disparity = req.equalized_odds.get("tpr_disparity", 0)
    eo_fpr_disparity = req.equalized_odds.get("fpr_disparity", 0)

    prompt = (
        "You are an ML fairness engineer designing practical mitigation strategies. "
        f"The sensitive attribute '{col}' has a fairness score of {req.fairness_score}/100. "
        f"\nDisparity metrics:\n"
        f"  - Demographic Parity disparity: {dp_disparity:.2%}\n"
        f"  - TPR disparity: {eo_tpr_disparity:.2%}\n"
        f"  - FPR disparity: {eo_fpr_disparity:.2%}\n"
        "\n"
        "Recommend 4 practical mitigation strategies with clear trade-offs. "
        "Prioritize strategies that address the highest disparities. "
        "Return ONLY a valid JSON array of 4 strategy objects (no markdown, no code blocks). "
        "Each object MUST have exactly these fields:\n"
        "- name (string): short strategy name\n"
        "- description (string): 2-3 sentences explaining the approach and when to use it\n"
        "- difficulty (Easy|Medium|Hard): implementation complexity\n"
        "- impact (Low|Medium|High): expected fairness improvement\n"
        "- code_hint (string): one Python import/method line that implements this strategy\n"
        "- tradeoff (string): accuracy loss or runtime cost implications"
    )

    if client:
        try:
            response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
            strategies = parse_gemini_json(response.text)
            return JSONResponse(content={"source": "gemini", "strategies": strategies})
        except Exception as e:
            print(f"Gemini mitigation error: {e}")

    # Fallback
    return JSONResponse(content={"source": "fallback", "strategies": [
        {"name": "Reweighing", "description": "Assign sample weights to balance group representation during training. This pre-processing approach requires no model architecture changes and works with any learner. Use when you have a clear training process.", "difficulty": "Easy", "impact": "High", "code_hint": f"from fairlearn.preprocessing import Reweighing; rw = Reweighing(prot_attr='{col}')", "tradeoff": "No accuracy loss expected; minimal runtime cost."},
        {"name": "Remove Sensitive Feature", "description": f"Drop '{col}' from training data entirely. Simple approach that prevents direct discrimination, but correlated features may still leak proxy information. Verify that related features don't recreate the bias.", "difficulty": "Easy", "impact": "Medium", "code_hint": f"X_train = X_train.drop(columns=['{col}'])", "tradeoff": "Risk of indirect discrimination if proxy features exist; may reduce model interpretability."},
        {"name": "Threshold Optimizer", "description": "Post-processing: adjust per-group decision thresholds to equalize TPR/FPR without retraining. Fast to implement and no changes to training code. Effective when false positive/negative rates differ by group.", "difficulty": "Medium", "impact": "High", "code_hint": "from fairlearn.postprocessing import ThresholdOptimizer; opt = ThresholdOptimizer(estimator=clf, constraints='equalized_odds')", "tradeoff": "May lower overall accuracy slightly; requires separate optimization threshold per group."},
        {"name": "Exponentiated Gradient", "description": "In-processing: constrained optimization during model training for fairness with minimal accuracy loss. Provides best accuracy-fairness trade-off and works with sklearn estimators. Recommended for production systems.", "difficulty": "Hard", "impact": "High", "code_hint": "from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds; eg = ExponentiatedGradient(estimator=clf, constraints=EqualizedOdds())", "tradeoff": "Requires retraining; higher computational cost; smaller fairness constraints trade off accuracy differently."}
    ]})
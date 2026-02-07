"""
PharmaNexus AI - Live Drug Approval & Safety Intelligence Platform
===================================================================
A regulatory AI system that simulates:
- Drug approval probability prediction (LightGBM)
- CDSCO-style regulatory decision logic
- Real-time post-market safety signal streaming (Kafka-like)
- Explainable AI outputs (SHAP)

All in ONE file: Backend + ML + Streaming + HTML UI

Run with: python pharmanexus_ai.py
Then open: http://127.0.0.1:8000
"""

import asyncio
import json
import random
import time
from datetime import datetime
from threading import Thread
from typing import Dict, List

import numpy as np
import pandas as pd
import shap
import uvicorn
import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel



app = FastAPI(title="PharmaNexus AI")


MODEL = None
EXPLAINER = None
FEATURE_NAMES = None


SAFETY_STREAM = {
    "events": [],
    "current_risk_level": "Low",
    "rolling_risk_score": 0.2,
    "last_update": datetime.now().isoformat(),
    "is_streaming": True,
    "has_cardio_hepato_events": False
}


def load_pretrained_model():
    """Load pre-trained XGBoost model and scaler from joblib files."""
    global MODEL, FEATURE_NAMES
    
    print(" Loading pre-trained model and scaler...")
    
    # Load model metadata (optional)
    metadata = {}
    try:
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(" model_metadata.json not found (optional file)")
    except json.JSONDecodeError:
        print(" model_metadata.json is invalid, ignoring")
    
    
    MODEL = joblib.load('drug_model.joblib')
    
    # Load scaler (optional)
    try:
        scaler = joblib.load('scaler.joblib')
    except FileNotFoundError:
        print(" scaler.joblib not found (optional)")
        scaler = None
    
   
    FEATURE_NAMES = metadata.get('features', [
        'tpsa', 'n_rings', 'n_rotatable_bonds', 'num_of_atoms',
        'num_of_heavy_atoms', 'num_heteroatoms', 'h_donors', 
        'h_acceptors', 'gastiger_charges'
    ])
    
    print(f"✓ Model loaded: {metadata.get('model_type', 'XGBoost')}")
    print(f"✓ Features ({len(FEATURE_NAMES)}): {', '.join(FEATURE_NAMES)}")
    if 'approval_threshold' in metadata:
        print(f"✓ Approval threshold: {metadata['approval_threshold']}")
    if 'trained_on' in metadata:
        print(f"✓ Trained on: {metadata['trained_on']}")
    print(" SHAP explainer will be initialized on first prediction (optional)")
    
    return scaler


def get_explainer():
    """
    Lazily initialize and return SHAP explainer.
    SHAP is optional - if it fails, return None and use fallback feature importance.
    """
    global EXPLAINER
    
   
    if EXPLAINER == "FAILED":
        return None
    
 
    if EXPLAINER is None:
        try:
            print("⚙️ Initializing SHAP TreeExplainer (optional feature)...")
            EXPLAINER = shap.TreeExplainer(MODEL)
            print("✓ SHAP explainer ready")
        except Exception as e:
            
            print(f" SHAP initialization skipped: {type(e).__name__}")
            print("   Using feature importance fallback instead")
            EXPLAINER = "FAILED"
            return None
    
    return EXPLAINER


def get_feature_importance_fallback():
    """
    Fallback method to get feature importance when SHAP is not available.
    Uses the model's built-in feature importance.
    """
    try:
       
        if hasattr(MODEL, 'feature_importances_'):
            importances = MODEL.feature_importances_
        elif hasattr(MODEL, 'get_booster'):
           
            booster = MODEL.get_booster()
            importance_dict = booster.get_score(importance_type='gain')
            
            importances = []
            for feat in FEATURE_NAMES:
                importances.append(importance_dict.get(feat, 0.0))
            importances = np.array(importances)
        else:
            
            return None
        
      
        if importances.sum() > 0:
            importances = importances / importances.sum()
        
        return importances
    except Exception:
        return None


def get_feature_importance_as_shap(X):
    """
    Convert feature importance to SHAP-like attribution values.
    This is a fallback when SHAP is not available.
    """
    importances = get_feature_importance_fallback()
    
    if importances is not None:
        
        feature_values = X.iloc[0].values
        
      
        feature_values_normalized = np.abs(feature_values) / (np.abs(feature_values).max() + 1e-10)
        
        
        attributions = importances * feature_values_normalized
        
        
        shap_dict = {
            feat: float(attributions[i]) 
            for i, feat in enumerate(FEATURE_NAMES)
        }
    else:
       
        shap_dict = {feat: 1.0 / len(FEATURE_NAMES) for feat in FEATURE_NAMES}
    
    return shap_dict


def derive_features(toxicity_risk: float, molecular_complexity: float, trial_novelty: float) -> Dict:
    """
    Map user-friendly UI inputs to molecular descriptor features expected by the pre-trained model.
    The model expects: tpsa, n_rings, n_rotatable_bonds, num_of_atoms, num_of_heavy_atoms, 
                       num_heteroatoms, h_donors, h_acceptors, gastiger_charges
    """
    
    noise = lambda scale=1.0: np.random.normal(0, 0.02 * scale)
    
    
    
    
    tpsa = max(0, min(200, 80 + (1 - toxicity_risk) * 50 + noise(10)))
    
    
    n_rings = max(0, int(molecular_complexity * 5 + noise(0.5)))
    
    
    n_rotatable_bonds = max(0, int(molecular_complexity * 10 + trial_novelty * 3 + noise(1)))
    
    num_of_atoms = max(10, int(30 + molecular_complexity * 40 + noise(5)))
    
    num_of_heavy_atoms = max(5, int(num_of_atoms * (0.7 + toxicity_risk * 0.2) + noise(2)))
    
    
    num_heteroatoms = max(0, int(10 + toxicity_risk * 8 + noise(2)))
    
   
    h_donors = max(0, int(3 + (1 - toxicity_risk) * 4 + noise(0.5)))
    
   
    h_acceptors = max(0, int(5 + (1 - toxicity_risk) * 6 + noise(1)))
    
    gastiger_charges = -1 + toxicity_risk * 2 + molecular_complexity * 0.5 + noise(0.2)
    
    return {
        'tpsa': float(tpsa),
        'n_rings': int(n_rings),
        'n_rotatable_bonds': int(n_rotatable_bonds),
        'num_of_atoms': int(num_of_atoms),
        'num_of_heavy_atoms': int(num_of_heavy_atoms),
        'num_heteroatoms': int(num_heteroatoms),
        'h_donors': int(h_donors),
        'h_acceptors': int(h_acceptors),
        'gastiger_charges': float(gastiger_charges)
    }


def compute_confidence(approval_prob: float, effective_threshold: float) -> tuple:
    """
    Compute prediction confidence based on distance from decision boundary.
    Returns (confidence_score, confidence_label)
    """
   
    distance_from_threshold = abs(approval_prob - effective_threshold)
    
    
    defer_threshold = effective_threshold - 0.15
    
    if approval_prob >= effective_threshold:
      
        confidence_score = min(1.0, distance_from_threshold / 0.25)
    elif approval_prob >= defer_threshold:
       
        min_dist = min(abs(approval_prob - effective_threshold), 
                      abs(approval_prob - defer_threshold))
        confidence_score = min(0.65, min_dist / 0.15)
    else:
    
        confidence_score = min(1.0, abs(approval_prob - defer_threshold) / 0.25)
    
    
    if approval_prob > 0.85 or approval_prob < 0.15:
        confidence_score = min(1.0, confidence_score * 1.2)
    
 
    if confidence_score >= 0.75:
        confidence_label = "High"
    elif confidence_score >= 0.45:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"
    
    return confidence_score, confidence_label


def compute_counterfactuals(
    current_features: Dict,
    current_prob: float,
    current_decision: str,
    effective_threshold: float
) -> List[Dict]:
    """
    Compute counterfactual insights - minimum changes needed to flip decisions.
    """
    counterfactuals = []
    
    # Determine target probabilities based on current decision
    defer_threshold = effective_threshold - 0.15
    
    if "Restrict" in current_decision or "Reject" in current_decision:
        # Need to reach defer zone
        target_prob = defer_threshold + 0.05
        target_decision = "Defer - Additional Data Required"
    elif "Defer" in current_decision:
        # Need to reach approval
        target_prob = effective_threshold + 0.05
        target_decision = "Approve for Market"
    else:
        # Already approved - show what would cause deferral
        target_prob = effective_threshold - 0.05
        target_decision = "Defer - Additional Data Required"
    
    prob_gap = target_prob - current_prob
    
    # Test perturbations on key features
    feature_impacts = []
    
    # Updated to use actual molecular descriptors with appropriate ranges
    for feature_name in ['tpsa', 'n_rings', 'h_donors', 'h_acceptors', 'num_heteroatoms']:
        if feature_name not in current_features:
            continue
            
        current_val = current_features[feature_name]
        
        if feature_name == 'tpsa':
            
            test_values = [min(200, current_val + delta) for delta in [10, 20, 30, 40]]
            direction = "increase"
        elif feature_name in ['n_rings']:
            # Rings: moderate is best, test small changes (range 0-6)
            test_values = [max(0, int(current_val - delta)) for delta in [1, 2, 3]]
            direction = "reduce"
        elif feature_name in ['h_donors', 'h_acceptors']:
            # H-bond donors/acceptors: more is generally better (range 0-15)
            test_values = [min(15, int(current_val + delta)) for delta in [1, 2, 3, 4]]
            direction = "increase"
        elif feature_name == 'num_heteroatoms':
            # Heteroatoms: moderate levels, test reduction (range 0-20)
            test_values = [max(0, int(current_val - delta)) for delta in [1, 2, 3, 4]]
            direction = "reduce"
        else:
            continue
        
        # Test each perturbation
        for test_val in test_values:
            test_features = current_features.copy()
            test_features[feature_name] = test_val
            
            X_test = pd.DataFrame([test_features])
            test_prob = float(MODEL.predict_proba(X_test)[0, 1])
            
            prob_change = test_prob - current_prob
            
            # Check if this crosses the threshold
            if (prob_gap > 0 and test_prob >= target_prob) or \
               (prob_gap < 0 and test_prob <= target_prob):
                
                feature_impacts.append({
                    'feature': feature_name,
                    'current_value': current_val,
                    'target_value': test_val,
                    'change': test_val - current_val,
                    'prob_impact': prob_change,
                    'direction': direction
                })
                break  # Found minimum change for this feature
    
    # Sort by smallest absolute change
    feature_impacts.sort(key=lambda x: abs(x['change']))
    
    # Take top 3 most actionable changes
    for impact in feature_impacts[:3]:
        feature_display = impact['feature'].replace('_', ' ').replace('tpsa', 'TPSA').replace('n ', 'Num ').title()
        change_val = impact['change']
        prob_impact_pct = impact['prob_impact'] * 100
        
        # Format change text based on feature type
        if impact['feature'] in ['tpsa', 'gastiger_charges']:
            # Float values - show absolute change
            if impact['direction'] == "reduce":
                change_text = f"-{abs(change_val):.1f}"
            else:
                change_text = f"+{abs(change_val):.1f}"
        else:
            # Integer values - show as integers
            if impact['direction'] == "reduce":
                change_text = f"-{abs(int(change_val))}"
            else:
                change_text = f"+{abs(int(change_val))}"
        
        action = "Reduce" if impact['direction'] == "reduce" else "Increase"
        
        counterfactuals.append({
            'feature': f"{action} {feature_display}",
            'current_value': impact['current_value'],
            'target_value': impact['target_value'],
            'change': change_text,
            'impact': f"{'+' if prob_impact_pct >= 0 else ''}{prob_impact_pct:.1f}% approval probability",
            'decision_flip': target_decision
        })
    
    
    if SAFETY_STREAM["current_risk_level"] == "High":
        counterfactuals.append({
            'feature': "Reduce Post-Market Safety Risk",
            'current_value': SAFETY_STREAM["rolling_risk_score"],
            'target_value': 0.4,
            'change': "High → Medium",
            'impact': "~+5-8% approval probability",
            'decision_flip': "May enable approval"
        })
    
    return counterfactuals




def generate_safety_event():
    """Generate a realistic adverse event."""
    # Most events are general, cardio/hepato events are rare
    general_events = [
        "Mild Adverse Reaction",
        "Moderate Side Effect",
        "Drug Interaction Alert",
        "Allergic Reaction Cluster",
        "Efficacy Concern",
    ]
    
    serious_events = [
        "Severe Adverse Event",
        "Hepatotoxicity Signal",
        "Cardiotoxicity Warning",
        "Liver Enzyme Elevation",
        "Cardiac Arrhythmia Detected",
        "QT Prolongation Alert"
    ]
    
    
    if random.random() < 0.85:
        event_type = random.choice(general_events)
    else:
        event_type = random.choice(serious_events)
    
    severities = ["Low", "Medium", "High"]
    
    # Higher chance of high severity for cardio/hepato events
    if "Cardiotoxicity" in event_type or "Hepatotoxicity" in event_type or \
       "Cardiac" in event_type or "Liver" in event_type:
        severity = random.choices(severities, weights=[0.2, 0.3, 0.5])[0]
    else:
        severity = random.choices(severities, weights=[0.6, 0.3, 0.1])[0]
    
    event = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "severity": severity,
        "patient_count": random.randint(1, 50) if severity == "Low" else random.randint(10, 200),
        "region": random.choice(["North America", "Europe", "Asia-Pacific", "Latin America"]),
        "is_cardio_or_hepato": ("Cardiotoxicity" in event_type or "Hepatotoxicity" in event_type or 
                                "Cardiac" in event_type or "Liver" in event_type)
    }
    
    return event


def update_safety_stream():
    """Background thread that continuously generates safety events."""
    global SAFETY_STREAM
    
    print(" Safety stream started (simulating Kafka-like real-time events)")
    
    while SAFETY_STREAM["is_streaming"]:
        time.sleep(random.uniform(2, 5))  # Events every 2-5 seconds
        
        event = generate_safety_event()
        SAFETY_STREAM["events"].append(event)
        
        # Check for cardio/hepato events
        if event.get("is_cardio_or_hepato"):
            SAFETY_STREAM["has_cardio_hepato_events"] = True
        
        # Keep only last 20 events
        if len(SAFETY_STREAM["events"]) > 20:
            SAFETY_STREAM["events"] = SAFETY_STREAM["events"][-20:]
            
            # Reset cardio/hepato flag if no such events in recent 20
            recent_cardio_hepato = any(e.get("is_cardio_or_hepato", False) 
                                      for e in SAFETY_STREAM["events"])
            SAFETY_STREAM["has_cardio_hepato_events"] = recent_cardio_hepato
        
        # Update rolling risk score based on recent events
        recent_events = SAFETY_STREAM["events"][-5:]
        severity_scores = {"Low": 0.2, "Medium": 0.5, "High": 0.9}
        avg_severity = np.mean([severity_scores[e["severity"]] for e in recent_events])
        
        # Boost risk score if cardio/hepato events present
        if SAFETY_STREAM["has_cardio_hepato_events"]:
            avg_severity = min(1.0, avg_severity * 1.3)
        
        # Smooth the risk score
        SAFETY_STREAM["rolling_risk_score"] = (
            SAFETY_STREAM["rolling_risk_score"] * 0.7 + avg_severity * 0.3
        )
        
        # Classify risk level
        if SAFETY_STREAM["rolling_risk_score"] < 0.35:
            SAFETY_STREAM["current_risk_level"] = "Low"
        elif SAFETY_STREAM["rolling_risk_score"] < 0.65:
            SAFETY_STREAM["current_risk_level"] = "Medium"
        else:
            SAFETY_STREAM["current_risk_level"] = "High"
        
        SAFETY_STREAM["last_update"] = datetime.now().isoformat()


# ============================================================================
# API MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    toxicity_risk: float
    molecular_complexity: float
    trial_novelty: float
    cdsco_strictness: int = 50  # 0-100 scale
    compound_type: str = "dry"  # "dry" or "wet"


class CounterfactualInsight(BaseModel):
    feature: str
    current_value: float
    target_value: float
    change: str
    impact: str
    decision_flip: str


class PredictionResponse(BaseModel):
    approval_probability: float
    regulatory_decision: str
    decision_color: str
    shap_values: Dict[str, float]
    user_inputs: Dict[str, float]
    derived_features: Dict[str, float]
    cdsco_threshold: float
    confidence_score: float
    confidence_label: str
    counterfactuals: List[CounterfactualInsight]
    compound_type: str
    requires_feedback: bool


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
   
    # Derive all features from user inputs
    all_features = derive_features(
        request.toxicity_risk,
        request.molecular_complexity,
        request.trial_novelty
    )
    
    # Adjust features based on compound type
    if request.compound_type == "wet":
        # Wet compounds typically have higher TPSA and more H-bond acceptors/donors
        all_features['tpsa'] = min(200, all_features['tpsa'] * 1.15)
        all_features['h_acceptors'] = int(min(15, all_features['h_acceptors'] * 1.2))
        all_features['h_donors'] = int(min(10, all_features['h_donors'] * 1.2))
    else:  # dry
        # Dry compounds typically have lower TPSA and fewer polar interactions
        all_features['tpsa'] = all_features['tpsa'] * 0.85
        all_features['h_acceptors'] = int(all_features['h_acceptors'] * 0.9)
        all_features['h_donors'] = int(all_features['h_donors'] * 0.9)
    
    # Prepare input for model
    X = pd.DataFrame([all_features])
    
    # Get prediction probability
    approval_prob = float(MODEL.predict_proba(X)[0, 1])
    
    # Calculate SHAP values (optional - use fallback if unavailable)
    explainer = get_explainer()
    
    if explainer is not None:
        # SHAP is available - use it
        try:
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Get positive class SHAP values
            
            shap_dict = {
                feat: float(shap_values[0, i]) 
                for i, feat in enumerate(FEATURE_NAMES)
            }
        except Exception as e:
            # SHAP calculation failed - use fallback
            print(f" SHAP calculation failed, using feature importance fallback")
            shap_dict = get_feature_importance_as_shap(X)
    else:
        # SHAP not available - use feature importance fallback
        shap_dict = get_feature_importance_as_shap(X)
    
    # Apply CDSCO strictness to threshold
    base_threshold = 0.5
    strictness_adjustment = (request.cdsco_strictness - 50) / 100
    cdsco_threshold = base_threshold + strictness_adjustment * 0.3
    
    # Factor in current safety stream risk
    safety_penalty = SAFETY_STREAM["rolling_risk_score"] * 0.1
    effective_threshold = cdsco_threshold + safety_penalty
    
    # Check for cardiotoxicity or hepatotoxicity events
    has_cardio_hepato = SAFETY_STREAM["has_cardio_hepato_events"]
    
    # Regulatory decision logic with safety blocking
    requires_feedback = False
    
    if has_cardio_hepato:
        # CRITICAL: Block approval if cardiotoxicity or hepatotoxicity detected
        decision = "BLOCKED - Cardiotoxicity/Hepatotoxicity Detected in Stream"
        color = "#DC2626"  # Dark red
        # Don't cap probability - show actual model prediction with warning
    elif approval_prob >= effective_threshold:
        if SAFETY_STREAM["current_risk_level"] == "High":
            decision = "Conditional Approval - User Feedback Required"
            color = "#F59E0B"  # Orange
            requires_feedback = True
        else:
            decision = "Approve for Market"
            color = "#10B981"  # Green
            # Still require feedback if borderline approval
            if approval_prob < effective_threshold + 0.1:
                requires_feedback = True
                decision = "Approve - Post-Market Monitoring Required"
    elif approval_prob >= effective_threshold - 0.15:
        decision = "Defer - Additional Data Required"
        color = "#F59E0B"  # Yellow
    else:
        decision = "Restrict / Reject"
        color = "#EF4444"  # Red
    
    # Compute confidence score
    confidence_score, confidence_label = compute_confidence(approval_prob, effective_threshold)
    
    # Compute counterfactual insights
    counterfactuals = compute_counterfactuals(
        all_features, 
        approval_prob, 
        decision, 
        effective_threshold
    )
    
    # Separate user inputs vs derived features  
    # Note: User inputs are kept as-is for UI, but model uses molecular descriptors
    user_inputs = {
        'toxicity_risk': request.toxicity_risk,
        'molecular_complexity': request.molecular_complexity,
        'trial_novelty': request.trial_novelty
    }
    
    # Calculate ADMET properties for the UI (Auto-Inferred Risk Signals)
    # These are the 4 essential features the frontend displays
    
    # Solubility (based on TPSA and lipophilicity)
    tpsa_norm = all_features['tpsa'] / 200.0  # Normalize TPSA
    solubility = min(1.0, max(0.0, tpsa_norm * 0.7 + (1 - request.toxicity_risk) * 0.3))
    
    # Bioavailability (Lipinski's rule approximation)
    # Good bioavailability if: MW < 500, logP < 5, H-donors <= 5, H-acceptors <= 10
    mw_score = 1.0 if all_features['num_of_atoms'] * 12 < 500 else 0.5
    hbd_score = 1.0 if all_features['h_donors'] <= 5 else 0.5
    hba_score = 1.0 if all_features['h_acceptors'] <= 10 else 0.5
    bioavailability = min(1.0, (mw_score + hbd_score + hba_score) / 3.0 * (1 - request.toxicity_risk * 0.3))
    
    # Clearance Risk (metabolic stability - inversely related to complexity)
    complexity_factor = all_features['n_rings'] / 6.0 + all_features['n_rotatable_bonds'] / 15.0
    clearance_risk = min(1.0, max(0.0, complexity_factor * 0.5 + request.toxicity_risk * 0.5))
    
    # ADMET Composite Score (overall drug-likeness)
    admet_composite = min(1.0, max(0.0, (
        solubility * 0.25 + 
        bioavailability * 0.35 + 
        (1 - clearance_risk) * 0.25 +
        (1 - request.toxicity_risk) * 0.15
    )))
    
    derived_features = {
        'solubility': round(solubility, 3),
        'bioavailability': round(bioavailability, 3),
        'clearance_risk': round(clearance_risk, 3),
        'admet_composite': round(admet_composite, 3)
    }
    
    # Add derived features to SHAP values for explainability
    # Approximate SHAP contributions based on feature values and impact
    shap_dict['solubility'] = (solubility - 0.5) * 0.15  # Positive solubility helps approval
    shap_dict['bioavailability'] = (bioavailability - 0.5) * 0.20  # High importance
    shap_dict['clearance_risk'] = -(clearance_risk - 0.5) * 0.12  # High risk hurts approval
    shap_dict['admet_composite'] = (admet_composite - 0.5) * 0.18  # Overall drug-likeness
    
    return PredictionResponse(
        approval_probability=approval_prob,
        regulatory_decision=decision,
        decision_color=color,
        shap_values=shap_dict,
        user_inputs=user_inputs,
        derived_features=derived_features,
        cdsco_threshold=effective_threshold,
        confidence_score=confidence_score,
        confidence_label=confidence_label,
        counterfactuals=counterfactuals,
        compound_type=request.compound_type,
        requires_feedback=requires_feedback
    )


@app.get("/stream_status")
async def get_stream_status():
    """Get current safety stream status."""
    return {
        "current_risk_level": SAFETY_STREAM["current_risk_level"],
        "rolling_risk_score": round(SAFETY_STREAM["rolling_risk_score"], 3),
        "recent_events": SAFETY_STREAM["events"][-5:],  # Last 5 events
        "last_update": SAFETY_STREAM["last_update"],
        "total_events_tracked": len(SAFETY_STREAM["events"])
    }


@app.get("/health")
async def health_check():
    """System health check."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "explainer_ready": EXPLAINER is not None,
        "stream_active": SAFETY_STREAM["is_streaming"]
    }


# ============================================================================
# HTML FRONTEND 
# ============================================================================

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PharmaNexus AI - Drug Approval Intelligence</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e3a8a 100%);
            background-attachment: fixed;
            min-height: 100vh;
            padding: 20px;
            position: relative;
            overflow-x: hidden;
        }
        
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 50%, rgba(139, 92, 246, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(59, 130, 246, 0.15) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }
        
        .container {
            max-width: 1800px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.98);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            box-shadow: 
                0 20px 60px rgba(0, 0, 0, 0.3),
                0 0 0 1px rgba(255, 255, 255, 0.1) inset;
            overflow: hidden;
            position: relative;
            z-index: 1;
        }
        
        /* Header */
        .header {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50% , #2563eb 100%);
            color: white;
            padding: 40px 50px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: relative;
            overflow: hidden;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -10%;
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
            border-radius: 50%;
        }
        
        .header::after {
            content: '';
            position: absolute;
            bottom: -30%;
            left: -5%;
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(139, 92, 246, 0.2) 0%, transparent 70%);
            border-radius: 50%;
        }
        
        .header-content {
            position: relative;
            z-index: 1;
        }
        
        .header h1 {
            font-size: 2.8em;
            font-weight: 800;
            letter-spacing: -1px;
            background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }
        
        .header .subtitle {
            font-size: 1.05em;
            opacity: 0.95;
            font-weight: 400;
            letter-spacing: 0.3px;
        }
        
        .status-badges {
            display: flex;
            gap: 12px;
            position: relative;
            z-index: 1;
        }
        
        .badge {
            padding: 10px 20px;
            border-radius: 30px;
            font-size: 0.9em;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        
        .badge:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        }
        
        .badge.success {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.9), rgba(5, 150, 105, 0.9));
        }
        
        .badge.warning {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.9), rgba(217, 119, 6, 0.9));
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: white;
            box-shadow: 0 0 10px rgba(255, 255, 255, 0.8);
            animation: pulse 2s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { 
                opacity: 1; 
                transform: scale(1);
            }
            50% { 
                opacity: 0.6; 
                transform: scale(0.9);
            }
        }
        
        /* Main Grid */
        .main-grid {
            display: grid;
            grid-template-columns: 380px 1fr 380px;
            gap: 30px;
            padding: 30px;
        }
        
        /* Panels */
        .panel {
            background: linear-gradient(135deg, rgba(248, 250, 252, 0.95) 0%, rgba(255, 255, 255, 0.95) 100%);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 32px;
            box-shadow: 
                0 4px 20px rgba(0, 0, 0, 0.06),
                0 0 0 1px rgba(0, 0, 0, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.8);
            transition: all 0.3s ease;
        }
        
        .panel:hover {
            box-shadow: 
                0 8px 30px rgba(0, 0, 0, 0.08),
                0 0 0 1px rgba(59, 130, 246, 0.1);
        }
        
        .panel-title {
            font-size: 1.4em;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 24px;
            padding-bottom: 12px;
            border-bottom: 3px solid #e5e7eb;
            display: flex;
            align-items: center;
            gap: 10px;
            letter-spacing: -0.3px;
        }
        
        .panel-title::before {
            content: '';
            width: 4px;
            height: 24px;
            background: linear-gradient(180deg, #3b82f6, #8b5cf6);
            border-radius: 2px;
        }
        
        /* Left Panel - Inputs */
        .input-group {
            margin-bottom: 28px;
        }
        
        .input-label {
            display: block;
            font-weight: 600;
            color: #334155;
            margin-bottom: 10px;
            font-size: 0.95em;
            letter-spacing: 0.2px;
        }
        
        .slider-container {
            position: relative;
            margin-bottom: 8px;
        }
        
        input[type="range"] {
            width: 100%;
            height: 10px;
            border-radius: 10px;
            background: linear-gradient(90deg, #e5e7eb, #cbd5e1);
            outline: none;
            -webkit-appearance: none;
            transition: all 0.3s ease;
        }
        
        input[type="range"]:hover {
            background: linear-gradient(90deg, #cbd5e1, #94a3b8);
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            cursor: pointer;
            box-shadow: 
                0 4px 12px rgba(59, 130, 246, 0.4),
                0 0 0 4px rgba(59, 130, 246, 0.1);
            transition: all 0.3s ease;
        }
        
        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.15);
            box-shadow: 
                0 6px 16px rgba(59, 130, 246, 0.5),
                0 0 0 6px rgba(59, 130, 246, 0.15);
        }
        
        input[type="range"]::-webkit-slider-thumb:active {
            transform: scale(1.05);
        }
        
        input[type="range"]::-moz-range-thumb {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            cursor: pointer;
            border: none;
            box-shadow: 
                0 4px 12px rgba(59, 130, 246, 0.4),
                0 0 0 4px rgba(59, 130, 246, 0.1);
            transition: all 0.3s ease;
        }
        
        input[type="range"]::-moz-range-thumb:hover {
            transform: scale(1.15);
        }
        
        .slider-value {
            text-align: right;
            font-size: 0.9em;
            color: #3b82f6;
            font-weight: 700;
            font-family: 'SF Mono', Monaco, monospace;
        }
        
        .compound-type-selector {
            background: linear-gradient(135deg, #eff6ff, #dbeafe);
            padding: 18px;
            border-radius: 12px;
            border: 2px solid #3b82f6;
            margin-top: 10px;
        }
        
        .compound-type-buttons {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 12px;
        }
        
        .compound-btn {
            padding: 12px 16px;
            border: 2px solid #cbd5e1;
            background: white;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.9em;
            cursor: pointer;
            transition: all 0.3s ease;
            color: #475569;
        }
        
        .compound-btn:hover {
            border-color: #3b82f6;
            background: #f0f9ff;
        }
        
        .compound-btn.active {
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            border-color: #1d4ed8;
            color: white;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        }
        
        .compound-description {
            font-size: 0.8em;
            color: #1e40af;
            font-weight: 500;
            line-height: 1.4;
        }
        
        .derived-section {
            margin-top: 36px;
            padding-top: 24px;
            border-top: 3px dashed #e5e7eb;
        }
        
        .derived-title {
            font-size: 1.05em;
            font-weight: 700;
            color: #6366f1;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .progress-bar-container {
            margin-bottom: 18px;
        }
        
        .progress-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.85em;
            color: #475569;
            margin-bottom: 8px;
            font-weight: 600;
        }
        
        .progress-bar {
            width: 100%;
            height: 14px;
            background: linear-gradient(90deg, #f1f5f9, #e2e8f0);
            border-radius: 10px;
            overflow: hidden;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.06);
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #6366f1, #8b5cf6, #a855f7);
            border-radius: 10px;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 0 10px rgba(139, 92, 246, 0.5);
        }
        
        .run-button {
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, #10B981, #059669);
            color: white;
            border: none;
            border-radius: 14px;
            font-size: 1.15em;
            font-weight: 700;
            cursor: pointer;
            margin-top: 28px;
            box-shadow: 
                0 6px 20px rgba(16, 185, 129, 0.35),
                0 0 0 1px rgba(255, 255, 255, 0.1) inset;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            letter-spacing: 0.3px;
            text-transform: uppercase;
        }
        
        .run-button:hover {
            transform: translateY(-3px);
            box-shadow: 
                0 10px 30px rgba(16, 185, 129, 0.45),
                0 0 0 1px rgba(255, 255, 255, 0.2) inset;
        }
        
        .run-button:active {
            transform: translateY(-1px);
        }
        
        .run-button.loading {
            background: linear-gradient(135deg, #6b7280, #4b5563);
            cursor: not-allowed;
            animation: loadingPulse 1.5s ease-in-out infinite;
        }
        
        @keyframes loadingPulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }
        
        /* Center Panel */
        .intelligence-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin: 24px 0;
        }
        
        .gauge-container {
            text-align: center;
        }
        
        .gauge {
            width: 220px;
            height: 220px;
            margin: 0 auto;
            position: relative;
            filter: drop-shadow(0 10px 30px rgba(0, 0, 0, 0.15));
        }
        
        .gauge-circle {
            width: 100%;
            height: 100%;
            border-radius: 50%;
            background: conic-gradient(
                from 0deg,
                #e5e7eb 0deg 360deg
            );
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            transition: background 0.6s ease-in-out;
        }
        
        .gauge-inner {
            width: 82%;
            height: 82%;
            background: linear-gradient(135deg, #ffffff, #f8fafc);
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: 
                0 8px 20px rgba(0, 0, 0, 0.1) inset,
                0 0 0 1px rgba(255, 255, 255, 0.8);
        }
        
        .gauge-value {
            font-size: 2.8em;
            font-weight: 800;
            background: linear-gradient(135deg, #1e293b, #475569);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -1px;
        }
        
        .gauge-label {
            font-size: 0.85em;
            color: #64748b;
            margin-top: 8px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .decision-card {
            background: linear-gradient(135deg, #ffffff, #f8fafc);
            border-radius: 16px;
            padding: 28px;
            margin: 24px 0;
            box-shadow: 
                0 8px 25px rgba(0, 0, 0, 0.08),
                0 0 0 1px rgba(0, 0, 0, 0.03);
            border-left: 6px solid #64748b;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .decision-card:hover {
            transform: translateX(4px);
            box-shadow: 
                0 12px 35px rgba(0, 0, 0, 0.12),
                0 0 0 1px rgba(59, 130, 246, 0.1);
        }
        
        .decision-text {
            font-size: 1.35em;
            font-weight: 700;
            color: #0f172a;
            letter-spacing: -0.3px;
        }
        
        .cdsco-slider-section {
            margin: 32px 0;
            padding: 24px;
            background: linear-gradient(135deg, #fff7ed, #fffbeb);
            border-radius: 14px;
            border: 2px solid #fdba74;
            box-shadow: 0 4px 15px rgba(251, 191, 36, 0.15);
        }
        
        .cdsco-title {
            font-size: 1.15em;
            font-weight: 700;
            color: #c2410c;
            margin-bottom: 12px;
            letter-spacing: -0.2px;
        }
        
        .cdsco-description {
            font-size: 0.9em;
            color: #9a3412;
            margin-bottom: 16px;
            font-weight: 500;
        }
        
        /* Right Panel - Safety Stream */
        .stream-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
        }
        
        .risk-indicator {
            padding: 10px 18px;
            border-radius: 24px;
            font-weight: 700;
            font-size: 0.9em;
            letter-spacing: 0.3px;
            text-transform: uppercase;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        
        .risk-low {
            background: linear-gradient(135deg, #d1fae5, #a7f3d0);
            color: #065f46;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }
        
        .risk-medium {
            background: linear-gradient(135deg, #fef3c7, #fde68a);
            color: #92400e;
            box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
        }
        
        .risk-high {
            background: linear-gradient(135deg, #fee2e2, #fecaca);
            color: #991b1b;
            animation: pulseRisk 2s ease-in-out infinite;
            box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
        }
        
        @keyframes pulseRisk {
            0%, 100% { 
                transform: scale(1); 
                box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
            }
            50% { 
                transform: scale(1.05); 
                box-shadow: 0 6px 18px rgba(239, 68, 68, 0.5);
            }
        }
        
        .event-feed {
            max-height: 450px;
            overflow-y: auto;
            padding-right: 12px;
        }
        
        .event-item {
            background: linear-gradient(135deg, #ffffff, #f8fafc);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            border-left: 4px solid #e5e7eb;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .event-item:hover {
            transform: translateX(4px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
            border-left-color: #3b82f6;
        }
        
        .event-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .event-type {
            font-weight: 700;
            font-size: 0.9em;
            color: #0f172a;
            letter-spacing: -0.1px;
        }
        
        .severity-badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        }
        
        .severity-low {
            background: linear-gradient(135deg, #d1fae5, #a7f3d0);
            color: #065f46;
        }
        
        .severity-medium {
            background: linear-gradient(135deg, #fef3c7, #fde68a);
            color: #92400e;
        }
        
        .severity-high {
            background: linear-gradient(135deg, #fee2e2, #fecaca);
            color: #991b1b;
        }
        
        .event-details {
            font-size: 0.8em;
            color: #64748b;
            margin-top: 8px;
            font-weight: 500;
        }
        
        /* Feedback Monitoring Section */
        .feedback-section {
            margin-top: 24px;
            padding-top: 24px;
            border-top: 3px solid #e5e7eb;
        }
        
        .feedback-header {
            font-size: 1.1em;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 16px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e5e7eb;
        }
        
        .feedback-status {
            background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
            padding: 14px;
            border-radius: 10px;
            margin-bottom: 14px;
            border: 2px solid #3b82f6;
        }
        
        .feedback-count {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        
        .count-label {
            font-size: 0.85em;
            color: #475569;
            font-weight: 600;
        }
        
        .count-value {
            font-size: 0.95em;
            color: #1e40af;
            font-weight: 700;
            font-family: 'SF Mono', Monaco, monospace;
        }
        
        .feedback-result {
            font-size: 0.9em;
            font-weight: 600;
            padding: 8px 12px;
            border-radius: 8px;
            margin-top: 8px;
        }
        
        .feedback-result.warning {
            background: linear-gradient(135deg, #fee2e2, #fecaca);
            color: #991b1b;
            border-left: 4px solid #ef4444;
            animation: pulseWarning 2s ease-in-out infinite;
        }
        
        .feedback-result.safe {
            background: linear-gradient(135deg, #d1fae5, #a7f3d0);
            color: #065f46;
            border-left: 4px solid #10b981;
        }
        
        @keyframes pulseWarning {
            0%, 100% { 
                transform: scale(1);
                box-shadow: 0 2px 8px rgba(239, 68, 68, 0.2);
            }
            50% { 
                transform: scale(1.02);
                box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
            }
        }
        
        .feedback-list {
            max-height: 180px;
            overflow-y: auto;
        }
        
        .feedback-item {
            background: white;
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 8px;
            border-left: 3px solid #cbd5e1;
            font-size: 0.85em;
        }
        
        .feedback-item.adverse {
            border-left-color: #ef4444;
            background: #fef2f2;
        }
        
        .feedback-item.normal {
            border-left-color: #10b981;
            background: #f0fdf4;
        }
        
        .feedback-user {
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 4px;
        }
        
        .feedback-text {
            color: #475569;
            line-height: 1.4;
        }
        
        /* Bottom Panel - Explainability */
        .explainability-panel {
            grid-column: 1 / -1;
        }
        
        .shap-section {
            margin-top: 20px;
        }
        
        .shap-category {
            margin-bottom: 25px;
        }
        
        .shap-category-title {
            font-size: 1em;
            font-weight: 600;
            color: #475569;
            margin-bottom: 15px;
            padding-left: 5px;
        }
        
        .shap-bar-container {
            margin-bottom: 12px;
        }
        
        .shap-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.85em;
            color: #475569;
            margin-bottom: 5px;
        }
        
        .shap-bar-wrapper {
            position: relative;
            height: 30px;
            background: #f1f5f9;
            border-radius: 5px;
            overflow: hidden;
        }
        
        .shap-bar {
            position: absolute;
            height: 100%;
            transition: all 0.5s ease;
        }
        
        .shap-bar.positive {
            background: linear-gradient(90deg, #10B981, #059669);
            right: 50%;
        }
        
        .shap-bar.negative {
            background: linear-gradient(90deg, #EF4444, #DC2626);
            left: 50%;
        }
        
        .center-line {
            position: absolute;
            left: 50%;
            top: 0;
            bottom: 0;
            width: 2px;
            background: #94a3b8;
        }
        
        .loading-text {
            text-align: center;
            color: #64748b;
            font-style: italic;
            margin: 20px 0;
        }
        
        /* Scrollbar styling */
        .event-feed::-webkit-scrollbar {
            width: 6px;
        }
        
        .event-feed::-webkit-scrollbar-track {
            background: #f1f5f9;
            border-radius: 3px;
        }
        
        .event-feed::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 3px;
        }
        
        .event-feed::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }
        
        /* Market Graphs Section */
        .market-section {
            background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
            padding: 40px;
            margin: 0 30px 30px 30px;
            border-radius: 20px;
            box-shadow: 
                0 4px 20px rgba(0, 0, 0, 0.08),
                0 0 0 1px rgba(0, 0, 0, 0.03);
            position: relative;
        }
        
        .market-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
            border-radius: 20px 20px 0 0;
        }
        
        .market-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e5e7eb;
        }
        
        .market-title {
            font-size: 1.6em;
            font-weight: 700;
            color: #0f172a;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .market-title::before {
            content: '';
            width: 6px;
            height: 32px;
            background: linear-gradient(180deg, #3b82f6, #8b5cf6);
            border-radius: 3px;
        }
        
        .market-timestamp {
            font-size: 0.95em;
            color: #64748b;
            font-family: 'SF Mono', Monaco, 'Courier New', monospace;
            font-weight: 600;
            background: #f1f5f9;
            padding: 8px 16px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }
        
        .market-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 24px;
        }
        
        .market-card {
            background: white;
            border-radius: 16px;
            padding: 24px;
            border: 2px solid #e5e7eb;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .market-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 100%;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.03) 0%, transparent 100%);
            opacity: 0;
            transition: opacity 0.4s ease;
        }
        
        .market-card:hover {
            transform: translateY(-5px) scale(1.02);
            box-shadow: 
                0 12px 40px rgba(0, 0, 0, 0.12),
                0 0 0 1px rgba(59, 130, 246, 0.1);
            border-color: #3b82f6;
        }
        
        .market-card:hover::before {
            opacity: 1;
        }
        
        .market-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
            position: relative;
            z-index: 1;
        }
        
        .company-name {
            font-weight: 700;
            color: #0f172a;
            font-size: 1.05em;
            letter-spacing: -0.2px;
        }
        
        .price-change {
            font-weight: 700;
            font-size: 0.95em;
            padding: 6px 14px;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        
        .price-change.positive {
            background: linear-gradient(135deg, #d1fae5, #a7f3d0);
            color: #065f46;
            box-shadow: 0 2px 8px rgba(16, 185, 129, 0.2);
        }
        
        .price-change.negative {
            background: linear-gradient(135deg, #fee2e2, #fecaca);
            color: #991b1b;
            box-shadow: 0 2px 8px rgba(239, 68, 68, 0.2);
        }
        
        .sentiment-badge {
            font-weight: 700;
            font-size: 0.9em;
            padding: 6px 14px;
            border-radius: 8px;
            background: linear-gradient(135deg, #e0e7ff, #c7d2fe);
            color: #3730a3;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.2);
        }
        
        .market-stats {
            display: flex;
            justify-content: space-around;
            margin-top: 18px;
            padding-top: 18px;
            border-top: 2px solid #f1f5f9;
            position: relative;
            z-index: 1;
        }
        
        .stat {
            text-align: center;
        }
        
        .stat-label {
            display: block;
            font-size: 0.8em;
            color: #64748b;
            margin-bottom: 6px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .stat-value {
            display: block;
            font-size: 1.1em;
            font-weight: 700;
            color: #0f172a;
        }
        
        canvas {
            display: block;
            margin: 12px auto;
            border-radius: 8px;
        }
        
        /* Confidence Card Styles */
        .confidence-container {
            display: flex;
            align-items: center;
        }
        
        .confidence-card {
            background: linear-gradient(135deg, #ffffff, #f8fafc);
            border-radius: 16px;
            padding: 28px;
            box-shadow: 
                0 8px 25px rgba(0, 0, 0, 0.08),
                0 0 0 1px rgba(0, 0, 0, 0.03);
            border: 3px solid #e5e7eb;
            text-align: center;
            width: 100%;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .confidence-card.high-confidence {
            border-color: #10B981;
            background: linear-gradient(135deg, #f0fdf4, #ffffff);
            box-shadow: 
                0 8px 25px rgba(16, 185, 129, 0.15),
                0 0 0 1px rgba(16, 185, 129, 0.1);
        }
        
        .confidence-card.medium-confidence {
            border-color: #F59E0B;
            background: linear-gradient(135deg, #fffbeb, #ffffff);
            box-shadow: 
                0 8px 25px rgba(245, 158, 11, 0.15),
                0 0 0 1px rgba(245, 158, 11, 0.1);
        }
        
        .confidence-card.low-confidence {
            border-color: #EF4444;
            background: linear-gradient(135deg, #fef2f2, #ffffff);
            box-shadow: 
                0 8px 25px rgba(239, 68, 68, 0.15),
                0 0 0 1px rgba(239, 68, 68, 0.1);
            animation: pulseConfidence 3s ease-in-out infinite;
        }
        
        @keyframes pulseConfidence {
            0%, 100% { 
                transform: scale(1); 
                box-shadow: 
                    0 8px 25px rgba(239, 68, 68, 0.15),
                    0 0 0 1px rgba(239, 68, 68, 0.1);
            }
            50% { 
                transform: scale(1.03); 
                box-shadow: 
                    0 12px 35px rgba(239, 68, 68, 0.25),
                    0 0 0 2px rgba(239, 68, 68, 0.2);
            }
        }
        
        .confidence-icon {
            font-size: 1.2em;
            margin-bottom: 12px;
            filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.1));
            font-weight: 800;
            color: #3b82f6;
            letter-spacing: 1px;
        }
        
        .confidence-label-text {
            font-size: 0.85em;
            color: #64748b;
            margin-bottom: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .confidence-value {
            font-size: 2em;
            font-weight: 800;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }
        
        .confidence-value.high {
            background: linear-gradient(135deg, #10B981, #059669);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .confidence-value.medium {
            background: linear-gradient(135deg, #F59E0B, #D97706);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .confidence-value.low {
            background: linear-gradient(135deg, #EF4444, #DC2626);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .confidence-score {
            font-size: 0.9em;
            color: #64748b;
            margin-bottom: 14px;
            font-weight: 600;
            font-family: 'SF Mono', Monaco, monospace;
        }
        
        .confidence-explanation {
            font-size: 0.8em;
            color: #475569;
            font-style: italic;
            margin-top: 14px;
            padding-top: 14px;
            border-top: 2px solid #e5e7eb;
            line-height: 1.5;
        }
        
        /* What-If Regulator Panel */
        .whatif-panel {
            grid-column: 1 / -1;
            background: linear-gradient(135deg, #fffbeb 0%, #fef9f5 100%);
            border: 3px solid #fbbf24;
            box-shadow: 0 6px 25px rgba(251, 191, 36, 0.2);
        }
        
        .whatif-panel.emphasized {
            animation: emphasizePulse 2s ease-in-out infinite;
            border-color: #f59e0b;
        }
        
        @keyframes emphasizePulse {
            0%, 100% { 
                box-shadow: 0 6px 25px rgba(251, 191, 36, 0.2); 
                transform: scale(1);
            }
            50% { 
                box-shadow: 0 10px 40px rgba(251, 191, 36, 0.35); 
                transform: scale(1.005);
            }
        }
        
        .whatif-subtitle {
            font-size: 1.05em;
            color: #92400e;
            margin-bottom: 24px;
            font-weight: 600;
            letter-spacing: 0.1px;
        }
        
        .whatif-subtitle.low-confidence-warning {
            color: #dc2626;
            font-weight: 700;
            padding: 14px 18px;
            background: linear-gradient(135deg, #fee2e2, #fecaca);
            border-radius: 10px;
            border-left: 5px solid #ef4444;
            box-shadow: 0 4px 15px rgba(239, 68, 68, 0.2);
        }
        
        .whatif-content {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 18px;
        }
        
        .counterfactual-card {
            background: white;
            border-radius: 14px;
            padding: 22px;
            border-left: 5px solid #3b82f6;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .counterfactual-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 100%;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.03) 0%, transparent 100%);
            opacity: 0;
            transition: opacity 0.4s ease;
        }
        
        .counterfactual-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(59, 130, 246, 0.15);
            border-left-color: #2563eb;
        }
        
        .counterfactual-card:hover::before {
            opacity: 1;
        }
        
        .counterfactual-feature {
            font-size: 1.05em;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 10px;
            position: relative;
            z-index: 1;
        }
        
        .counterfactual-feature::before {
            content: "→";
            font-size: 1.5em;
            color: #3b82f6;
            font-weight: 700;
        }
        
        .counterfactual-change {
            display: inline-block;
            background: linear-gradient(135deg, #dbeafe, #bfdbfe);
            color: #1e40af;
            padding: 6px 16px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.95em;
            margin: 10px 0;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.2);
            position: relative;
            z-index: 1;
        }
        
        .counterfactual-impact {
            font-size: 0.9em;
            color: #475569;
            margin-top: 10px;
            position: relative;
            z-index: 1;
            font-weight: 500;
        }
        
        .counterfactual-impact strong {
            color: #10B981;
            font-weight: 700;
        }
        
        .counterfactual-decision {
            margin-top: 12px;
            padding: 10px 14px;
            background: linear-gradient(135deg, #f0fdf4, #dcfce7);
            border-radius: 8px;
            font-size: 0.85em;
            color: #065f46;
            font-weight: 700;
            position: relative;
            z-index: 1;
            box-shadow: 0 2px 8px rgba(16, 185, 129, 0.15);
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="header-content">
                <h1>PharmaNexus AI</h1>
                <div class="subtitle">Live Drug Approval & Safety Intelligence Platform</div>
            </div>
            <div class="status-badges">
                <div class="badge success">
                    <div class="status-dot"></div>
                    Model Loaded
                </div>
                <div class="badge warning">
                    <div class="status-dot"></div>
                    Streaming Active
                </div>
            </div>
        </div>
        
        <!-- Market Graphs Section -->
        <div class="market-section">
            <div class="market-header">
                <div class="market-title">Live Pharma Market Intelligence</div>
                <div class="market-timestamp" id="marketTimestamp">--</div>
            </div>
            <div class="market-grid">
                <div class="market-card">
                    <div class="market-card-header">
                        <span class="company-name">Global Pharma Index</span>
                        <span class="price-change positive" id="index-change">+0.00%</span>
                    </div>
                    <canvas id="indexChart" width="300" height="120"></canvas>
                    <div class="market-stats">
                        <div class="stat">
                            <span class="stat-label">Current</span>
                            <span class="stat-value" id="index-current">$0.00</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Volume</span>
                            <span class="stat-value" id="index-volume">0M</span>
                        </div>
                    </div>
                </div>
                
                <div class="market-card">
                    <div class="market-card-header">
                        <span class="company-name">BioTech Leaders ETF</span>
                        <span class="price-change negative" id="biotech-change">+0.00%</span>
                    </div>
                    <canvas id="biotechChart" width="300" height="120"></canvas>
                    <div class="market-stats">
                        <div class="stat">
                            <span class="stat-label">Current</span>
                            <span class="stat-value" id="biotech-current">$0.00</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Volume</span>
                            <span class="stat-value" id="biotech-volume">0M</span>
                        </div>
                    </div>
                </div>
                
                <div class="market-card">
                    <div class="market-card-header">
                        <span class="company-name">Drug Approval Sentiment</span>
                        <span class="sentiment-badge" id="sentiment-badge">Neutral</span>
                    </div>
                    <canvas id="sentimentChart" width="300" height="120"></canvas>
                    <div class="market-stats">
                        <div class="stat">
                            <span class="stat-label">Approvals (30d)</span>
                            <span class="stat-value" id="approvals-count">0</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Rejections</span>
                            <span class="stat-value" id="rejections-count">0</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Main Grid -->
        <div class="main-grid">
            <!-- Left Panel - Inputs -->
            <div class="panel">
                <div class="panel-title">Input Parameters</div>
                
                <div class="input-group">
                    <label class="input-label">Toxicity Risk</label>
                    <div class="slider-container">
                        <input type="range" id="toxicity" min="0" max="100" value="30" step="1">
                    </div>
                    <div class="slider-value" id="toxicity-value">0.30</div>
                </div>
                
                <div class="input-group">
                    <label class="input-label">Molecular Complexity</label>
                    <div class="slider-container">
                        <input type="range" id="complexity" min="0" max="100" value="50" step="1">
                    </div>
                    <div class="slider-value" id="complexity-value">0.50</div>
                </div>
                
                <div class="input-group">
                    <label class="input-label">Trial Novelty Score</label>
                    <div class="slider-container">
                        <input type="range" id="novelty" min="0" max="100" value="60" step="1">
                    </div>
                    <div class="slider-value" id="novelty-value">0.60</div>
                </div>
                
                <div class="input-group compound-type-selector">
                    <label class="input-label">Compound Classification</label>
                    <div class="compound-type-buttons">
                        <button class="compound-btn active" id="dry-compound" onclick="selectCompoundType('dry')">
                            Dry Compound
                        </button>
                        <button class="compound-btn" id="wet-compound" onclick="selectCompoundType('wet')">
                            Wet Compound
                        </button>
                    </div>
                    <div class="compound-description" id="compound-description">
                        Dry: Solid formulations (tablets, capsules, powders)
                    </div>
                </div>
                
                <!-- Derived Features -->
                <div class="derived-section">
                    <div class="derived-title">Auto-Inferred Risk Signals</div>
                    
                    <div class="progress-bar-container">
                        <div class="progress-label">
                            <span>Solubility</span>
                            <span id="solubility-val">--</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="solubility-bar" style="width: 0%"></div>
                        </div>
                    </div>
                    
                    <div class="progress-bar-container">
                        <div class="progress-label">
                            <span>Bioavailability</span>
                            <span id="bioavailability-val">--</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="bioavailability-bar" style="width: 0%"></div>
                        </div>
                    </div>
                    
                    <div class="progress-bar-container">
                        <div class="progress-label">
                            <span>Clearance Risk</span>
                            <span id="clearance-val">--</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="clearance-bar" style="width: 0%"></div>
                        </div>
                    </div>
                    
                    <div class="progress-bar-container">
                        <div class="progress-label">
                            <span>ADMET Composite</span>
                            <span id="admet-val">--</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="admet-bar" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
                
                <button class="run-button" id="runButton" onclick="runPrediction()">
                    Run Prediction
                </button>
            </div>
            
            <!-- Center Panel - Core Intelligence -->
            <div class="panel">
                <div class="panel-title">Approval Intelligence</div>
                
                <div class="intelligence-grid">
                    <div class="gauge-container">
                        <div class="gauge">
                            <div class="gauge-circle">
                                <div class="gauge-inner">
                                    <div class="gauge-value" id="gaugeValue">--</div>
                                    <div class="gauge-label">Approval Probability</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="confidence-container">
                        <div class="confidence-card" id="confidenceCard">
                            <div class="confidence-icon" id="confidenceIcon">TARGET</div>
                            <div class="confidence-label-text">Decision Confidence</div>
                            <div class="confidence-value" id="confidenceValue">--</div>
                            <div class="confidence-score" id="confidenceScore">--</div>
                            <div class="confidence-explanation" id="confidenceExplanation">
                                Run prediction to see confidence metrics
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="decision-card" id="decisionCard">
                    <div class="decision-text" id="decisionText">
                        Run prediction to see regulatory decision
                    </div>
                </div>
                
                <div class="cdsco-slider-section">
                    <div class="cdsco-title">CDSCO Regulatory Strictness</div>
                    <div class="cdsco-description">
                        Adjust regulatory conservativeness in real-time
                    </div>
                    <div class="input-group">
                        <input type="range" id="cdsco" min="0" max="100" value="50" step="1">
                        <div class="slider-value" id="cdsco-value">50% (Moderate)</div>
                    </div>
                </div>
            </div>
            
            <!-- Right Panel - Safety Stream -->
            <div class="panel">
                <div class="stream-header">
                    <div class="panel-title" style="margin: 0;">Live Safety Stream</div>
                    <div class="risk-indicator" id="riskIndicator">
                        Low Risk
                    </div>
                </div>
                
                <div class="event-feed" id="eventFeed">
                    <div class="loading-text">Loading safety events...</div>
                </div>
                
                <div class="feedback-section" id="feedbackSection" style="display: none;">
                    <div class="feedback-header">User Feedback Monitoring</div>
                    <div class="feedback-status" id="feedbackStatus">
                        <div class="feedback-count">
                            <span class="count-label">Responses:</span>
                            <span class="count-value" id="responseCount">0/3</span>
                        </div>
                        <div class="feedback-result" id="feedbackResult"></div>
                    </div>
                    <div class="feedback-list" id="feedbackList"></div>
                </div>
            </div>
            
            <!-- What-If Regulator Panel -->
            <div class="panel whatif-panel">
                <div class="panel-title">What-If Regulator - Counterfactual Decision Intelligence</div>
                <div class="whatif-subtitle" id="whatifSubtitle">
                    What would regulators need to see to change this decision?
                </div>
                
                <div class="whatif-content" id="whatifContent">
                    <div class="loading-text">Run a prediction to see counterfactual insights</div>
                </div>
            </div>
            
            <!-- Bottom Panel - Explainability -->
            <div class="panel explainability-panel">
                <div class="panel-title">Model Explainability - Why This Decision?</div>
                
                <div class="shap-section" id="shapSection">
                    <div class="loading-text">Run a prediction to see feature contributions</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // =====================================================================
        // MARKET GRAPHS - Live Data Simulation
        // =====================================================================
        
        // Market data storage
        const marketData = {
            index: {
                data: [],
                basePrice: 1250.0,
                current: 1250.0,
                change: 0
            },
            biotech: {
                data: [],
                basePrice: 485.0,
                current: 485.0,
                change: 0
            },
            sentiment: {
                data: [],
                approvals: 0,
                rejections: 0
            }
        };
        
        // Initialize market data
        function initializeMarketData() {
            const now = Date.now();
            for (let i = 50; i >= 0; i--) {
                const timestamp = now - i * 2000; // 2 second intervals
                
                // Index data
                const indexPrice = marketData.index.basePrice + (Math.random() - 0.5) * 20;
                marketData.index.data.push({ time: timestamp, value: indexPrice });
                
                // Biotech data
                const biotechPrice = marketData.biotech.basePrice + (Math.random() - 0.5) * 15;
                marketData.biotech.data.push({ time: timestamp, value: biotechPrice });
                
                // Sentiment data
                const sentimentScore = 50 + (Math.random() - 0.5) * 30;
                marketData.sentiment.data.push({ time: timestamp, value: sentimentScore });
            }
            
            marketData.index.current = marketData.index.data[marketData.index.data.length - 1].value;
            marketData.biotech.current = marketData.biotech.data[marketData.biotech.data.length - 1].value;
            marketData.sentiment.approvals = Math.floor(Math.random() * 15) + 5;
            marketData.sentiment.rejections = Math.floor(Math.random() * 8) + 2;
        }
        
        // Update market data with new tick
        function updateMarketData() {
            const now = Date.now();
            
            // Index - random walk with slight upward bias
            const indexChange = (Math.random() - 0.48) * 3;
            const newIndexPrice = marketData.index.current + indexChange;
            marketData.index.data.push({ time: now, value: newIndexPrice });
            marketData.index.data = marketData.index.data.slice(-50);
            const indexPctChange = ((newIndexPrice - marketData.index.basePrice) / marketData.index.basePrice) * 100;
            marketData.index.current = newIndexPrice;
            marketData.index.change = indexPctChange;
            
            // Biotech - more volatile
            const biotechChange = (Math.random() - 0.5) * 4;
            const newBiotechPrice = marketData.biotech.current + biotechChange;
            marketData.biotech.data.push({ time: now, value: newBiotechPrice });
            marketData.biotech.data = marketData.biotech.data.slice(-50);
            const biotechPctChange = ((newBiotechPrice - marketData.biotech.basePrice) / marketData.biotech.basePrice) * 100;
            marketData.biotech.current = newBiotechPrice;
            marketData.biotech.change = biotechPctChange;
            
            // Sentiment - oscillates
            const currentSentiment = marketData.sentiment.data[marketData.sentiment.data.length - 1].value;
            const sentimentChange = (Math.random() - 0.5) * 8;
            const newSentiment = Math.max(0, Math.min(100, currentSentiment + sentimentChange));
            marketData.sentiment.data.push({ time: now, value: newSentiment });
            marketData.sentiment.data = marketData.sentiment.data.slice(-50);
            
            // Occasionally update approval/rejection counts
            if (Math.random() > 0.95) {
                marketData.sentiment.approvals += Math.random() > 0.5 ? 1 : 0;
            }
            if (Math.random() > 0.97) {
                marketData.sentiment.rejections += 1;
            }
        }
        
        // Draw line chart on canvas
        function drawLineChart(canvasId, data, color, fillColor) {
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;
            
            const ctx = canvas.getContext('2d');
            const width = canvas.width;
            const height = canvas.height;
            
            // Clear canvas
            ctx.clearRect(0, 0, width, height);
            
            if (data.length < 2) return;
            
            // Calculate min/max for scaling
            const values = data.map(d => d.value);
            const minValue = Math.min(...values);
            const maxValue = Math.max(...values);
            const range = maxValue - minValue || 1;
            
            // Draw grid lines
            ctx.strokeStyle = '#e2e8f0';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {
                const y = (height / 4) * i;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }
            
            // Draw area fill
            ctx.beginPath();
            ctx.moveTo(0, height);
            
            data.forEach((point, i) => {
                const x = (width / (data.length - 1)) * i;
                const y = height - ((point.value - minValue) / range) * height;
                if (i === 0) {
                    ctx.lineTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            
            ctx.lineTo(width, height);
            ctx.closePath();
            ctx.fillStyle = fillColor;
            ctx.fill();
            
            // Draw line
            ctx.beginPath();
            data.forEach((point, i) => {
                const x = (width / (data.length - 1)) * i;
                const y = height - ((point.value - minValue) / range) * height;
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.5;
            ctx.stroke();
        }
        
        // Update market UI
        function updateMarketUI() {
            // Update timestamp
            const now = new Date();
            document.getElementById('marketTimestamp').textContent = 
                now.toLocaleTimeString() + ' (Live)';
            
            // Index
            const indexChange = document.getElementById('index-change');
            indexChange.textContent = (marketData.index.change >= 0 ? '+' : '') + 
                marketData.index.change.toFixed(2) + '%';
            indexChange.className = 'price-change ' + 
                (marketData.index.change >= 0 ? 'positive' : 'negative');
            document.getElementById('index-current').textContent = 
                '$' + marketData.index.current.toFixed(2);
            document.getElementById('index-volume').textContent = 
                (Math.random() * 50 + 150).toFixed(1) + 'M';
            drawLineChart('indexChart', marketData.index.data, '#10B981', 'rgba(16, 185, 129, 0.1)');
            
            // Biotech
            const biotechChange = document.getElementById('biotech-change');
            biotechChange.textContent = (marketData.biotech.change >= 0 ? '+' : '') + 
                marketData.biotech.change.toFixed(2) + '%';
            biotechChange.className = 'price-change ' + 
                (marketData.biotech.change >= 0 ? 'positive' : 'negative');
            document.getElementById('biotech-current').textContent = 
                '$' + marketData.biotech.current.toFixed(2);
            document.getElementById('biotech-volume').textContent = 
                (Math.random() * 30 + 80).toFixed(1) + 'M';
            drawLineChart('biotechChart', marketData.biotech.data, '#3b82f6', 'rgba(59, 130, 246, 0.1)');
            
            // Sentiment
            const currentSentiment = marketData.sentiment.data[marketData.sentiment.data.length - 1].value;
            const sentimentBadge = document.getElementById('sentiment-badge');
            if (currentSentiment > 65) {
                sentimentBadge.textContent = 'Bullish';
                sentimentBadge.style.background = '#d1fae5';
                sentimentBadge.style.color = '#065f46';
            } else if (currentSentiment > 35) {
                sentimentBadge.textContent = 'Neutral';
                sentimentBadge.style.background = '#e0e7ff';
                sentimentBadge.style.color = '#3730a3';
            } else {
                sentimentBadge.textContent = 'Bearish';
                sentimentBadge.style.background = '#fee2e2';
                sentimentBadge.style.color = '#991b1b';
            }
            
            document.getElementById('approvals-count').textContent = marketData.sentiment.approvals;
            document.getElementById('rejections-count').textContent = marketData.sentiment.rejections;
            drawLineChart('sentimentChart', marketData.sentiment.data, '#8b5cf6', 'rgba(139, 92, 246, 0.1)');
        }
        
        // =====================================================================
        // ORIGINAL CODE - Slider value updates
        // =====================================================================
        
        let selectedCompoundType = 'dry';
        let feedbackCollectionActive = false;
        let userFeedbacks = [];
        
        function selectCompoundType(type) {
            selectedCompoundType = type;
            
            // Update button states
            document.getElementById('dry-compound').classList.toggle('active', type === 'dry');
            document.getElementById('wet-compound').classList.toggle('active', type === 'wet');
            
            // Update description
            const descriptions = {
                dry: 'Dry: Solid formulations (tablets, capsules, powders)',
                wet: 'Wet: Liquid formulations (solutions, suspensions, syrups)'
            };
            document.getElementById('compound-description').textContent = descriptions[type];
        }
        
        const sliders = {
            toxicity: document.getElementById('toxicity'),
            complexity: document.getElementById('complexity'),
            novelty: document.getElementById('novelty'),
            cdsco: document.getElementById('cdsco')
        };
        
        sliders.toxicity.addEventListener('input', (e) => {
            document.getElementById('toxicity-value').textContent = (e.target.value / 100).toFixed(2);
        });
        
        sliders.complexity.addEventListener('input', (e) => {
            document.getElementById('complexity-value').textContent = (e.target.value / 100).toFixed(2);
        });
        
        sliders.novelty.addEventListener('input', (e) => {
            document.getElementById('novelty-value').textContent = (e.target.value / 100).toFixed(2);
        });
        
        sliders.cdsco.addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            let label = 'Moderate';
            if (val < 33) label = 'Lenient';
            else if (val > 66) label = 'Strict';
            document.getElementById('cdsco-value').textContent = `${val}% (${label})`;
        });
        
        // Run prediction
        async function runPrediction() {
            const button = document.getElementById('runButton');
            button.classList.add('loading');
            button.textContent = 'Running model inference...';
            
            const data = {
                toxicity_risk: parseFloat(sliders.toxicity.value) / 100,
                molecular_complexity: parseFloat(sliders.complexity.value) / 100,
                trial_novelty: parseFloat(sliders.novelty.value) / 100,
                cdsco_strictness: parseInt(sliders.cdsco.value),
                compound_type: selectedCompoundType
            };
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                // Update gauge
                const prob = (result.approval_probability * 100).toFixed(1);
                document.getElementById('gaugeValue').textContent = prob + '%';
                updateGaugeVisual(result.approval_probability);
                
                // Update decision card
                const decisionCard = document.getElementById('decisionCard');
                decisionCard.style.borderLeftColor = result.decision_color;
                document.getElementById('decisionText').textContent = result.regulatory_decision;
                
                // Update confidence card
                updateConfidenceCard(result.confidence_label, result.confidence_score);
                
                // Update derived features
                updateDerivedFeatures(result.derived_features);
                
                // Update counterfactuals
                updateCounterfactuals(result.counterfactuals, result.confidence_label);
                
                // Update SHAP values
                updateShapValues(result.shap_values, result.user_inputs, result.derived_features);
                
                // Handle feedback requirement
                if (result.requires_feedback) {
                    initiateFeedbackCollection();
                } else {
                    hideFeedbackSection();
                }
                
            } catch (error) {
                console.error('Prediction error:', error);
            } finally {
                button.classList.remove('loading');
                button.textContent = 'Run Prediction';
            }
        }
        
        function initiateFeedbackCollection() {
            feedbackCollectionActive = true;
            userFeedbacks = [];
            
            const feedbackSection = document.getElementById('feedbackSection');
            feedbackSection.style.display = 'block';
            
            document.getElementById('responseCount').textContent = '0/3';
            document.getElementById('feedbackResult').textContent = 'Awaiting user feedback...';
            document.getElementById('feedbackResult').className = 'feedback-result';
            document.getElementById('feedbackList').innerHTML = '';
            
            // Simulate user feedback over time
            simulateUserFeedback();
        }
        
        function hideFeedbackSection() {
            document.getElementById('feedbackSection').style.display = 'none';
            feedbackCollectionActive = false;
        }
        
        function updateGaugeVisual(probability) {
            // Convert probability (0-1) to degrees (0-360)
            // The gauge is a full circle, we'll show the probability as a sweep
            const degrees = probability * 360;
            
            // Create a gradient that shows:
            // - Red for 0-33% (0-120deg)
            // - Yellow for 33-66% (120-240deg)  
            // - Green for 66-100% (240-360deg)
            
            const gaugeCircle = document.querySelector('.gauge-circle');
            
            // Determine the color based on probability
            let gradient;
            if (probability <= 0.33) {
                // Low probability - show red up to the value, then gray
                gradient = `conic-gradient(
                    from 0deg,
                    #EF4444 0deg ${degrees}deg,
                    #e5e7eb ${degrees}deg 360deg
                )`;
            } else if (probability <= 0.66) {
                // Medium probability - show red, then yellow up to value, then gray
                gradient = `conic-gradient(
                    from 0deg,
                    #EF4444 0deg 120deg,
                    #F59E0B 120deg ${degrees}deg,
                    #e5e7eb ${degrees}deg 360deg
                )`;
            } else {
                // High probability - show red, yellow, then green up to value, then gray if not 100%
                if (degrees < 360) {
                    gradient = `conic-gradient(
                        from 0deg,
                        #EF4444 0deg 120deg,
                        #F59E0B 120deg 240deg,
                        #10B981 240deg ${degrees}deg,
                        #e5e7eb ${degrees}deg 360deg
                    )`;
                } else {
                    // Full circle at 100%
                    gradient = `conic-gradient(
                        from 0deg,
                        #EF4444 0deg 120deg,
                        #F59E0B 120deg 240deg,
                        #10B981 240deg 360deg
                    )`;
                }
            }
            
            gaugeCircle.style.background = gradient;
        }
        
        function simulateUserFeedback() {
            const feedbackScenarios = [
                { user: 'User #1', hasEffects: false, text: 'No adverse effects observed. Feeling normal.' },
                { user: 'User #2', hasEffects: false, text: 'Product working as expected, no issues.' },
                { user: 'User #3', hasEffects: false, text: 'No side effects, satisfied with results.' },
                { user: 'User #1', hasEffects: true, text: 'Experiencing mild nausea and dizziness.' },
                { user: 'User #2', hasEffects: true, text: 'Noticed elevated heart rate and discomfort.' },
                { user: 'User #3', hasEffects: true, text: 'Severe headache and fatigue after consumption.' }
            ];
            
            // Randomly decide if there will be adverse effects
            const hasAdverseEffects = Math.random() > 0.6; // 40% chance of adverse effects
            
            let feedbackIndex = 0;
            const feedbackInterval = setInterval(() => {
                if (feedbackIndex >= 3) {
                    clearInterval(feedbackInterval);
                    finalizeFeedback(hasAdverseEffects);
                    return;
                }
                
                const feedback = hasAdverseEffects ? 
                    feedbackScenarios[feedbackIndex + 3] : 
                    feedbackScenarios[feedbackIndex];
                
                userFeedbacks.push(feedback);
                
                // Add feedback to list
                const feedbackList = document.getElementById('feedbackList');
                const feedbackItem = document.createElement('div');
                feedbackItem.className = 'feedback-item ' + (feedback.hasEffects ? 'adverse' : 'normal');
                feedbackItem.innerHTML = `
                    <div class="feedback-user">${feedback.user}</div>
                    <div class="feedback-text">${feedback.text}</div>
                `;
                feedbackList.appendChild(feedbackItem);
                
                feedbackIndex++;
                document.getElementById('responseCount').textContent = `${feedbackIndex}/3`;
                
            }, 3000); // New feedback every 3 seconds
        }
        
        function finalizeFeedback(hasAdverseEffects) {
            const resultEl = document.getElementById('feedbackResult');
            
            if (hasAdverseEffects) {
                resultEl.textContent = 'WARNING: Adverse effects reported! Market withdrawal recommended.';
                resultEl.className = 'feedback-result warning';
                
                // Show warning in decision card
                const decisionCard = document.getElementById('decisionCard');
                decisionCard.style.borderLeftColor = '#ef4444';
                document.getElementById('decisionText').textContent = 
                    'MARKET WITHDRAWAL REQUIRED - Adverse User Effects Detected';
            } else {
                resultEl.textContent = 'All users report normal effects. Product safe for continued market presence.';
                resultEl.className = 'feedback-result safe';
            }
        }
        
        function updateConfidenceCard(label, score) {
            const card = document.getElementById('confidenceCard');
            const valueEl = document.getElementById('confidenceValue');
            const scoreEl = document.getElementById('confidenceScore');
            const explanationEl = document.getElementById('confidenceExplanation');
            
            // Remove old classes
            card.classList.remove('high-confidence', 'medium-confidence', 'low-confidence');
            valueEl.classList.remove('high', 'medium', 'low');
            
            // Add new classes
            const labelLower = label.toLowerCase();
            card.classList.add(labelLower + '-confidence');
            valueEl.classList.add(labelLower);
            
            // Update content
            valueEl.textContent = label.toUpperCase();
            scoreEl.textContent = `Confidence Score: ${(score * 100).toFixed(1)}%`;
            
            // Update explanation based on confidence level
            if (label === 'High') {
                explanationEl.textContent = 'Strong prediction - model is confident in this decision';
            } else if (label === 'Medium') {
                explanationEl.textContent = 'Moderate confidence - decision is relatively stable';
            } else {
                explanationEl.textContent = '⚠️ Low confidence - high uncertainty detected. Human review strongly recommended. See counterfactual analysis below.';
            }
        }
        
        function updateCounterfactuals(counterfactuals, confidenceLabel) {
            const contentEl = document.getElementById('whatifContent');
            const subtitleEl = document.getElementById('whatifSubtitle');
            const panelEl = document.querySelector('.whatif-panel');
            
            // Emphasize panel if low confidence
            if (confidenceLabel === 'Low') {
                panelEl.classList.add('emphasized');
                subtitleEl.className = 'whatif-subtitle low-confidence-warning';
                subtitleEl.textContent = '⚠️ COUNTERFACTUAL ANALYSIS RECOMMENDED: High uncertainty detected - explore decision boundaries';
            } else {
                panelEl.classList.remove('emphasized');
                subtitleEl.className = 'whatif-subtitle';
                subtitleEl.textContent = 'What would regulators need to see to change this decision?';
            }
            
            if (!counterfactuals || counterfactuals.length === 0) {
                contentEl.innerHTML = '<div class="loading-text">No counterfactual changes needed - decision is stable</div>';
                return;
            }
            
            let html = '';
            counterfactuals.forEach(cf => {
                html += `
                    <div class="counterfactual-card">
                        <div class="counterfactual-feature">${cf.feature}</div>
                        <div class="counterfactual-change">${cf.change}</div>
                        <div class="counterfactual-impact">
                            Impact: <strong>${cf.impact}</strong>
                        </div>
                        <div class="counterfactual-decision">
                            ✓ Decision flip: ${cf.decision_flip}
                        </div>
                    </div>
                `;
            });
            
            contentEl.innerHTML = html;
        }
        
        function updateDerivedFeatures(features) {
            const featureMap = {
                solubility: 'solubility',
                bioavailability: 'bioavailability',
                clearance_risk: 'clearance',
                admet_composite: 'admet'
            };
            
            for (const [key, id] of Object.entries(featureMap)) {
                const value = features[key];
                const percent = (value * 100).toFixed(1);
                document.getElementById(`${id}-val`).textContent = value.toFixed(3);
                document.getElementById(`${id}-bar`).style.width = percent + '%';
            }
        }
        
        function updateShapValues(shapValues, userInputs, derivedFeatures) {
            const shapSection = document.getElementById('shapSection');
            
            let html = '<div class="shap-category"><div class="shap-category-title"> User Input Contributions</div>';
            for (const [feature, value] of Object.entries(shapValues)) {
                if (feature in userInputs) {
                    html += createShapBar(feature, value);
                }
            }
            html += '</div>';
            
            html += '<div class="shap-category"><div class="shap-category-title"> Inferred Risk Signal Contributions</div>';
            for (const [feature, value] of Object.entries(shapValues)) {
                if (feature in derivedFeatures) {
                    html += createShapBar(feature, value);
                }
            }
            html += '</div>';
            
            shapSection.innerHTML = html;
        }
        
        function createShapBar(feature, value) {
            const absValue = Math.abs(value);
            const maxShap = 0.5; // Approximate max SHAP value
            const width = Math.min(100, (absValue / maxShap) * 100);
            const displayValue = value.toFixed(4);
            const sign = value >= 0 ? '+' : '';
            const className = value >= 0 ? 'positive' : 'negative';
            
            const featureName = feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            
            return `
                <div class="shap-bar-container">
                    <div class="shap-label">
                        <span>${featureName}</span>
                        <span style="font-weight: 600; color: ${value >= 0 ? '#10B981' : '#EF4444'}">${sign}${displayValue}</span>
                    </div>
                    <div class="shap-bar-wrapper">
                        <div class="center-line"></div>
                        <div class="shap-bar ${className}" style="width: ${width}%"></div>
                    </div>
                </div>
            `;
        }
        
        // Safety stream updates
        async function updateSafetyStream() {
            try {
                const response = await fetch('/stream_status');
                const data = await response.json();
                
                // Update risk indicator
                const riskIndicator = document.getElementById('riskIndicator');
                riskIndicator.textContent = data.current_risk_level + ' Risk';
                riskIndicator.className = 'risk-indicator risk-' + data.current_risk_level.toLowerCase();
                
                // Update event feed
                const eventFeed = document.getElementById('eventFeed');
                let html = '';
                
                for (const event of data.recent_events.reverse()) {
                    const time = new Date(event.timestamp).toLocaleTimeString();
                    html += `
                        <div class="event-item">
                            <div class="event-header">
                                <div class="event-type">${event.event_type}</div>
                                <div class="severity-badge severity-${event.severity.toLowerCase()}">${event.severity}</div>
                            </div>
                            <div class="event-details">
                                ${time} • ${event.patient_count} patients • ${event.region}
                            </div>
                        </div>
                    `;
                }
                
                eventFeed.innerHTML = html;
                
            } catch (error) {
                console.error('Stream update error:', error);
            }
        }
        
        // Start safety stream updates
        setInterval(updateSafetyStream, 3000);
        updateSafetyStream();
        
        // Initialize and start market graphs
        initializeMarketData();
        updateMarketUI();
        setInterval(() => {
            updateMarketData();
            updateMarketUI();
        }, 2000); // Update every 2 seconds for smooth live effect
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI."""
    return HTML_TEMPLATE


# ============================================================================
# STARTUP & MAIN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize model and start safety stream on server startup."""
    print("=" * 70)
    print("PharmaNexus AI - Drug Approval Intelligence Platform")
    print("=" * 70)
    
    # Load pre-trained model
    load_pretrained_model()
    
    # Start safety stream in background thread
    stream_thread = Thread(target=update_safety_stream, daemon=True)
    stream_thread.start()
    
    print("\n" + "=" * 70)
    print(" System ready!")
    print("=" * 70)
    print("\n Open http://127.0.0.1:8000 to view the live AI system.\n")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

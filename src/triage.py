# src/triage.py

import re
from typing import List, Dict, Any

# Symptom synonyms for matching
SYMPTOM_SYNONYMS = {
    "chest pain": ["chest ache", "chest discomfort", "chest tightness"],
    "dizziness": ["dizzy", "lightheaded", "faint"],
    "shortness of breath": ["breathlessness", "difficulty breathing"],
    "sweating": ["sweaty", "perspiration"],
    "nausea": ["sick to stomach", "queasy"],
    "palpitations": ["heart racing", "irregular heartbeat"],
    "headache": ["head pain", "migraine"],
    "heartburn": ["acid reflux", "burning in chest"],
}

# Triage rules
TRIAGE_RULES = [
    {
        "level": "URGENT",
        "conditions": [
            {
                "all_of": ["chest pain"],
                "any_of": ["shortness of breath", "sweating", "nausea", "palpitations"]
            },
            {
                "all_of": ["dizziness"],
                "any_of": ["palpitations", "shortness of breath", "chest pain"]
            }
        ]
    },
    {
        "level": "NON_URGENT",
        "conditions": [
            {
                "all_of": [],
                "any_of": []
            }
        ]
    }
]

def _match_symptom(text: str, symptom: str) -> bool:
    """Check if symptom or its synonyms are in the text."""
    text = text.lower()
    if symptom in SYMPTOM_SYNONYMS:
        phrases = [symptom.lower()] + [s.lower() for s in SYMPTOM_SYNONYMS[symptom]]
    else:
        phrases = [symptom.lower()]
    for phrase in phrases:
        if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text):
            return True
    return False

def simple_triage(predictions: List[str], premise: str) -> Dict[str, Any]:
    """Apply triage rules based on premise first, then predictions."""
    triage_result = {"level": "NON_URGENT", "reasons": []}

    # Check rules against premise (input text)
    for rule in TRIAGE_RULES:
        for condition in rule["conditions"]:
            all_met = all(_match_symptom(premise, s) for s in condition["all_of"])
            any_met = any(_match_symptom(premise, s) for s in condition["any_of"]) if condition["any_of"] else True
            if all_met and any_met:
                triage_result["level"] = rule["level"]
                triage_result["reasons"].append(condition)
                return triage_result  # Return first matching rule

    # Fallback to predictions if no premise-based rule matches
    for rule in TRIAGE_RULES:
        for condition in rule["conditions"]:
            all_met = all(s in predictions for s in condition["all_of"])
            any_met = any(s in predictions for s in condition["any_of"]) if condition["any_of"] else True
            if all_met and any_met:
                triage_result["level"] = rule["level"]
                triage_result["reasons"].append(condition)
                return triage_result

    return triage_result

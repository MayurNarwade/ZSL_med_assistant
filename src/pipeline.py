# src/pipeline.py
import os
import re
import json
import argparse
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .normalise import normalize
from .triage import simple_triage
from .recommend import recommend_specialists, recommend_tests

# -------- Paths --------
ROOT = os.path.dirname(os.path.dirname(__file__))
CONCEPTS = os.path.join(ROOT, "data", "concepts.jsonl")
INDEX = os.path.join(ROOT, "models", "faiss_index.bin")

# -------- Models (LIGHTWEIGHT VERSIONS) --------
EMB_MODEL = "intfloat/multilingual-e5-base"  
NLI_MODEL = "joeddav/xlm-roberta-large-xnli"  

# -------- Tunables --------
MAX_LEN = 512
RETR_SIM_MIN = 0.85
LEXICAL_BOOST = 0.35
MARGIN_MIN = 0.15
STRICT_EXACT_WINS = True
TOPK = 30

# -------- Globals --------
_enc = None
_index = None
_concepts = None
_tokenizer = None
_nli = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Loaders
# -------------------------
def _load_enc() -> SentenceTransformer:
    global _enc
    if _enc is None:
        _enc = SentenceTransformer(EMB_MODEL)
    return _enc

def _load_index():
    global _index
    if _index is None:
        if not os.path.exists(INDEX):
            raise FileNotFoundError(f"FAISS index not found at {INDEX}")
        _index = faiss.read_index(INDEX)
    return _index

def _load_concepts() -> List[Dict[str, Any]]:
    global _concepts
    if _concepts is None:
        if not os.path.exists(CONCEPTS):
            raise FileNotFoundError(f"Concepts file missing at {CONCEPTS}")
        with open(CONCEPTS, "r", encoding="utf-8") as f:
            _concepts = [json.loads(l) for l in f if l.strip()]
    return _concepts

def _load_nli():
    global _tokenizer, _nli
    if _nli is None:
        _tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL, use_fast=False)
        _tokenizer.model_max_length = MAX_LEN
        _nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(_device)
        _nli.eval()
    return _tokenizer, _nli

# -------------------------
# Lexical helper
# -------------------------
def _collect_phrases_for_concept(c: Dict[str, Any]) -> List[str]:
    phrases = []
    lab = c.get("label", "")
    if lab:
        phrases.append(lab.strip().lower())
    syns = c.get("synonyms", []) or []
    if isinstance(syns, str):
        syns = [s.strip() for s in syns.split("|") if s.strip()]
    for s in syns:
        if s:
            phrases.append(s.strip().lower())
    return list(dict.fromkeys(phrases))

def _lexical_hits(text: str, concepts: List[Dict[str, Any]]) -> Dict[int, float]:
    hits = {}
    t = text.lower()
    for idx, c in enumerate(concepts):
        phrases = _collect_phrases_for_concept(c)
        for p in phrases:
            if not p:
                continue
            pat = rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])"
            if re.search(pat, t):
                hits[idx] = max(hits.get(idx, 0.0), LEXICAL_BOOST)
    return hits

# -------------------------
# Embedding + Retrieval
# -------------------------
def _embed(texts: List[str]) -> np.ndarray:
    enc = _load_enc()
    vecs = enc.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vecs.astype("float32")

def _retrieve(user_text: str, k: int = TOPK, debug: bool = False):
    idx = _load_index()
    concepts = _load_concepts()

    qv = _embed([user_text])
    sims, ids = idx.search(qv, k)

    coarse = []
    seen = set()

    for i, s in zip(ids[0], sims[0]):
        if 0 <= i < len(concepts) and i not in seen:
            coarse.append((i, float(s)))
            seen.add(i)

    # lexical boost
    lex_hits = _lexical_hits(user_text, concepts)
    for i, boost in lex_hits.items():
        if i not in seen:
            coarse.append((i, 0.0))
            seen.add(i)

    if not coarse:
        return []

    enc = _load_enc()
    query_emb = enc.encode(user_text, convert_to_tensor=True, normalize_embeddings=True)

    cand_strings = []
    ids_order = []
    for cid, _ in coarse:
        c = concepts[cid]
        cand_strings.append(f"{c.get('label','')}. {c.get('description','')}")
        ids_order.append(cid)

    cand_embs = enc.encode(cand_strings, convert_to_tensor=True, normalize_embeddings=True)
    cos_scores = util.cos_sim(query_emb, cand_embs)[0].cpu().numpy()

    merged = []
    for cid, cos_score in zip(ids_order, cos_scores):
        total_score = float(cos_score) + float(lex_hits.get(cid, 0.0))
        merged.append((cid, total_score))

    merged.sort(key=lambda x: -x[1])
    return [(concepts[cid], score) for cid, score in merged]

# -------------------------
# NLI / Entailment
# -------------------------
def _entailment_full(premise: str, hypothesis: str):
    tok, nli = _load_nli()
    xs = tok(premise, hypothesis, return_tensors="pt",
             truncation=True, max_length=MAX_LEN).to(_device)
    with torch.no_grad():
        logits = nli(**xs).logits.softmax(-1)[0].cpu().numpy()

    # contradiction / neutral / entailment
    p_c, p_n, p_e = float(logits[0]), float(logits[1]), float(logits[2])
    margin = p_e - max(p_c, p_n)
    return p_e, margin

# -------------------------
# Inference
# -------------------------
def infer(text: str, k: int = TOPK, threshold: float = 0.75,
          topn: int = 5, debug: bool = False):

    premise = normalize(text)
    concepts = _load_concepts()
    cands = _retrieve(premise, k=k, debug=debug)

    scored = []
    for c, retr in cands:
        label = c.get("label", "")
        hyp = f"The patient has {label}."

        phrases = _collect_phrases_for_concept(c)
        lexical_present = any(
            re.search(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])",
                      premise.lower()) for p in phrases
        )

        if STRICT_EXACT_WINS and lexical_present:
            p_e, margin = 0.999, 0.999
        else:
            try:
                p_e, margin = _entailment_full(premise, hyp)
            except:
                p_e, margin = retr, 0.0

        if lexical_present or (p_e >= threshold and margin >= MARGIN_MIN):
            scored.append({
                "label": label,
                "score": p_e,
                "system": c.get("system", "general"),
                "desc": c.get("description", ""),
                "retrieval_sim": retr
            })

    scored.sort(key=lambda x: -x["score"])
    filtered = [s for s in scored if s["retrieval_sim"] >= RETR_SIM_MIN]
    keep = filtered[:topn]

    # enrich
    for p in keep:
        try:
            p["specialists"] = recommend_specialists(p["label"])
            p["recommended_tests"] = recommend_tests(p["label"])
        except:
            p["specialists"] = []
            p["recommended_tests"] = []

    triage = simple_triage([s["label"] for s in keep], premise)

    return {
        "normalized_text": premise,
        "predictions": keep,
        "triage": triage,
        "disclaimer": "Informational only; not a medical diagnosis."
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, required=True)
    args = ap.parse_args()

    out = infer(args.text)
    print(json.dumps(out, ensure_ascii=False, indent=2))

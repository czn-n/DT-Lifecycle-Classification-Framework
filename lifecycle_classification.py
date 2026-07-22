"""
Sentence-BERT based document classifier for Digital Twin literature in Civil Infrastructure.

Paper Reference:
Life Cycle-Oriented Digital Twin for Civil Infrastructure Assets:
A Natural Language Processing Model-Assisted Review
"""

import os
import re
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util


# Global Configurations & Constants

# Keywords classification dictionary
KEYS: Dict[str, Dict[str, List[str]]] = {
    "Design_Planning": {
        "strict": [
            "conceptual design", "preliminary design", "schematic design",
            "feasibility study", "design review", "design verification",
            "layout planning", "alignment planning", "generative design",
            "parametric design", "geometry optimization", "design alternative"
        ],
        "mid": [
            "bim authoring", "information modeling", "architectural design",
            "structural design", "design intent", "site planning"
        ],
        "broad": [
            "design", "planning", "planning phase", "design phase", "model authoring"
        ]
    },
    "Construction": {
        "strict": [
            "formwork", "rebar tying", "concrete pouring", "steel erection",
            "crane operation", "excavation", "backfilling", "scaffolding",
            "temporary works", "mep installation", "as-built", "earthwork"
        ],
        "mid": [
            "construction site", "site operations", "construction management",
            "progress monitoring", "prefabrication", "modular construction",
            "construction logistics", "quality inspection", "site inspection"
        ],
        "broad": [
            "construction", "construction phase", "onsite", "on-site", "site management",
            "resource allocation", "schedule management"
        ]
    },
    "Operation_Maintenance": {
        "strict": [
            "facility management", "asset management", "predictive maintenance",
            "preventive maintenance", "corrective maintenance", "structural health monitoring",
            "condition assessment", "condition monitoring", "fault detection", "damage detection"
        ],
        "mid": [
            "inspection", "maintenance schedule", "performance monitoring",
            "energy management", "occupancy detection", "sensor network",
            "remote monitoring", "operation phase"
        ],
        "broad": [
            "operation", "maintenance", "monitoring", "facility", "asset", "service life"
        ]
    },
    "Demolition_Renovation": {
        "strict": [
            "demolition", "dismantling", "decommissioning", "end-of-life", "demolition planning"
        ],
        "mid": [
            "retrofit", "retrofitting", "refurbishment", "reconstruction", "adaptive reuse"
        ],
        "broad": [
            "renovation", "upgrade", "renewal", "recycling", "rehabilitation"
        ]
    }
}

# Lifecycle prototype texts
STAGE_TEXTS: Dict[str, str] = {
    "Design_Planning": """
    This category includes studies where digital twins are primarily applied during the design and planning stage of infrastructure projects.
    Typical applications involve conceptual design, feasibility analysis, design optimization, planning support,
    Building Information Modeling (BIM), virtual prototyping, simulation-driven design,
    decision making before construction, risk assessment, lifecycle planning,
    and digital modeling for project planning and engineering design.
    design planning conceptual schematic feasibility layout parametric generative bim authoring architectural design
    """,
    "Construction": """
    This category includes studies where digital twins are primarily applied during the construction stage.
    Representative applications include construction monitoring, construction progress tracking,
    quality inspection, site management, resource allocation, schedule optimization,
    construction safety management, equipment monitoring, prefabrication,
    as-built modeling, construction simulation, and real-time construction control.
    construction site excavation formwork concrete rebar crane erection onsite progress monitoring construction management
    """,
    "Operation_Maintenance": """
    This category includes studies where digital twins are primarily applied during the operation and maintenance stage of infrastructure assets.
    Typical applications include structural health monitoring, condition assessment,
    predictive maintenance, preventive maintenance, asset management,
    infrastructure inspection, damage detection, anomaly detection,
    sensor data integration, performance monitoring, fault diagnosis,
    lifecycle management, maintenance decision support, and infrastructure operation.
    operation maintenance facility management asset structural health monitoring inspection condition monitoring performance monitoring
    """,
    "Demolition_Renovation": """
    This category includes studies where digital twins are primarily applied during the demolition, renovation, rehabilitation, or end-of-life stage of infrastructure assets.
    Representative applications include demolition planning, decommissioning,
    structural rehabilitation, retrofitting, refurbishment,
    adaptive reuse, reconstruction, renovation planning,
    recycling management, end-of-life assessment,
    sustainability evaluation, and lifecycle extension.
    demolition retrofit renovation decommissioning dismantling end-of-life refurbishment
    """
}

def word_boundary_count(
    text: str,
    phrase: str
) -> int:
    """
    Counts exact occurrences of a phrase in a given text using regex word boundaries.

    Args:
        text (str): Target text string.
        phrase (str): Keyword or phrase to match.

    Returns:
        int: Frequency of exact match occurrences.
    """
    if not phrase:
        return 0
    pattern = r"\b" + re.escape(phrase) + r"\b"
    return len(re.findall(pattern, text))


def compute_keyword_hits(
    text: str,
    weight_strict: float = 4.0,
    weight_mid: float = 2.0,
    weight_broad: float = 1.0
) -> Tuple[Dict[str, float], Dict[str, List[Tuple[str, str, int]]]]:
    """
    Computes keyword matching scores for each phase.

    Args:
        text (str): Normalized input text.
        weight_strict (float): Weight assigned to strict tier keywords.
        weight_mid (float): Weight assigned to mid tier keywords.
        weight_broad (float): Weight assigned to broad tier keywords.

    Returns:
        Tuple[Dict[str, float], Dict[str, List[Tuple[str, str, int]]]]:
            - Scores dictionary mapping lifecycle phase to score.
            - Matched dictionary detailing matched phrase, tier, and frequency.
    """
    scores: Dict[str, float] = defaultdict(float)
    matched: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)

    for stage, groups in KEYS.items():
        for p in groups.get("strict", []):
            c = word_boundary_count(text, p)
            if c:
                scores[stage] += weight_strict * c
                matched[stage].append((p, "strict", c))
        for p in groups.get("mid", []):
            c = word_boundary_count(text, p)
            if c:
                scores[stage] += weight_mid * c
                matched[stage].append((p, "mid", c))
        for p in groups.get("broad", []):
            c = word_boundary_count(text, p)
            if c:
                scores[stage] += weight_broad * c
                matched[stage].append((p, "broad", c))

    return scores, matched


def classify_document(
    text: str,
    model: SentenceTransformer,
    stage_embeddings: Dict[str, Any],
    weight_strict: float = 4.0,
    weight_mid: float = 2.0,
    weight_broad: float = 1.0,
    alpha: float = 3.0,
    low_conf_thresh: float = 0.25,
    margin_thresh: float = 0.1,
    score_scale: float = 3.5
) -> Dict[str, Any]:
    """
    Classifies a document text into one of the infrastructure lifecycle stages
    using a hybrid approach (keyword matching + SBERT semantic similarity).

    Identifies low-confidence predictions matching.

    Args:
        text (str): Raw document text.
        model (SentenceTransformer): Loaded Sentence-BERT model instance.
        stage_embeddings (Dict[str, Any]): Pre-computed stage prototype embeddings.
        weight_strict (float): Weight for strict keywords.
        weight_mid (float): Weight for mid-tier keywords.
        weight_broad (float): Weight for broad keywords.
        alpha (float): Multiplier weight for SBERT cosine similarity.
        low_conf_thresh (float): Confidence threshold for screening.
        margin_thresh (float): Threshold for score difference between top 2 phases.
        score_scale (float): Scaling reference score for sigmoid normalization.

    Returns:
        Dict[str, Any]: Classification details including predicted stage and confidence.
    """
    text_low = (text or "").lower()

    # keyword matching score
    kw_scores, kw_matched = compute_keyword_hits(
        text_low,
        weight_strict=weight_strict,
        weight_mid=weight_mid,
        weight_broad=weight_broad
    )

    # SBERT cosine similarity
    emb = model.encode(text_low, convert_to_tensor=True)
    emb_sims = {stage: float(util.cos_sim(emb, ref)) for stage, ref in stage_embeddings.items()}

    # combined score
    combined = {}
    for stage in stage_embeddings.keys():
        combined[stage] = float(kw_scores.get(stage, 0.0)) + alpha * float(emb_sims.get(stage, 0.0))

    # Assign record to phase with highest score
    sorted_stages = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    best_stage, top1_score = sorted_stages[0]
    _, top2_score = sorted_stages[1]

    # Calculate score difference between Top 1 and Top 2
    score_diff = top1_score - top2_score

    # Compute Absolute Normalized Confidence Score in [0, 1]
    confidence = float(1.0 / (1.0 + np.exp(-(top1_score - score_scale) / 2.0)))

    # Low-confidence decision:
    # Flagged if confidence < threshold OR score difference between top phases is small
    low_conf_flag = (confidence < low_conf_thresh) or (score_diff < margin_thresh)

    return {
        "predicted_stage": best_stage,
        "combined_scores": combined,
        "kw_matched": kw_matched,
        "emb_sims": emb_sims,
        "raw_best_score": top1_score,
        "score_diff_top2": score_diff,
        "confidence": confidence,
        "low_conf": low_conf_flag
    }


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Classify Digital Twin literature into lifecycle stages."
    )
    parser.add_argument(
        "--input_csv", type=str, required=True,
        help="Path to input CSV file containing literature records."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Directory where classification outputs and reports will be saved."
    )
    parser.add_argument(
        "--model_name", type=str, default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name or local path."
    )
    parser.add_argument(
        "--weight_strict", type=float, default=4.0,
        help="Weight for strict level keywords."
    )
    parser.add_argument(
        "--weight_mid", type=float, default=2.0,
        help="Weight for mid level keywords."
    )
    parser.add_argument(
        "--weight_broad", type=float, default=1.0,
        help="Weight for broad level keywords."
    )
    parser.add_argument(
        "--alpha", type=float, default=3.0,
        help="Weight multiplier alpha for SBERT similarity."
    )
    parser.add_argument(
        "--score_scale", type=float, default=3.5,
        help="Reference score for confidence sigmoid scaling."
    )
    parser.add_argument(
        "--low_conf_thresh", type=float, default=0.25,
        help="Confidence threshold for screening low-confidence records (e.g., 0.20, 0.25, 0.30)."
    )
    parser.add_argument(
        "--margin_thresh", type=float, default=0.5,
        help="Threshold for score difference between top 2 phases."
    )
    return parser.parse_args()



def main() -> None:
    """
    Main function executing the literature classification pipeline.
    """
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading input literature dataset from: {args.input_csv}")

    encodings_to_try = ["utf-8-sig", "gbk", "gb2312", "latin1", "utf-8"]
    df = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(args.input_csv, encoding=enc).fillna("")
            print(f"Successfully read CSV using encoding: {enc}")
            break
        except (UnicodeDecodeError, Exception):
            continue

    if df is None:
        raise ValueError(f"Could not read file {args.input_csv} with any common encodings.")

    # Construct unified text
    required_cols = {"Title", "Abstract", "Author Keywords", "Index Keywords"}
    if required_cols.issubset(df.columns):
        df["text"] = (
            df[["Title", "Abstract", "Author Keywords", "Index Keywords"]]
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )
    else:
        df["text"] = df.apply(lambda r: " ".join([str(x) for x in r.values]), axis=1).str.lower()

    # Load Model
    print(f"Loading SentenceTransformer model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    # Pre-compute target stage reference prototype embeddings
    print("Pre-computing stage reference prototype embeddings...")
    stage_embeddings = {
        stage: model.encode(text, convert_to_tensor=True)
        for stage, text in STAGE_TEXTS.items()
    }

    # Classification Loop
    records: List[Dict[str, Any]] = []
    texts = df["text"].tolist()
    total = len(texts)
    print(f"Classifying {total} document records...")

    for i, t in enumerate(texts):
        res = classify_document(
            text=t,
            model=model,
            stage_embeddings=stage_embeddings,
            weight_strict=args.weight_strict,
            weight_mid=args.weight_mid,
            weight_broad=args.weight_broad,
            alpha=args.alpha,
            low_conf_thresh=args.low_conf_thresh,
            margin_thresh=args.margin_thresh,
            score_scale=args.score_scale
        )
        records.append(res)
        if (i + 1) % 500 == 0 or i == total - 1:
            print(f"  Processed {i + 1}/{total} records")

    # Export Output Data
    rec_df = pd.DataFrame(records)
    out_df = pd.concat([df.reset_index(drop=True), rec_df.reset_index(drop=True)], axis=1)

    out_full = os.path.join(args.output_dir, "full_classification_results.csv")
    out_df.to_csv(out_full, index=False, encoding="utf-8-sig")

    # Export per-stage CSV files
    for stage in STAGE_TEXTS.keys():
        sub_df = out_df[out_df["predicted_stage"] == stage]
        if not sub_df.empty:
            sub_path = os.path.join(args.output_dir, f"{stage}.csv")
            sub_df.to_csv(sub_path, index=False, encoding="utf-8-sig")

    # Export low-confidence samples for manual review
    low_conf_df = out_df[out_df["low_conf"] == True]
    low_conf_path = os.path.join(args.output_dir, "low_confidence_samples.csv")
    low_conf_df.to_csv(low_conf_path, index=False, encoding="utf-8-sig")

    # Save summary report
    counts = out_df["predicted_stage"].value_counts().to_dict()
    low_conf_count = int(low_conf_df.shape[0])
    report = {
        "counts": counts,
        "total": total,
        "low_conf_count": low_conf_count,
        "low_conf_ratio": round(low_conf_count / total, 5) if total > 0 else 0.0
    }
    report_path = os.path.join(args.output_dir, "classification_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Total Processed: {total}")
    print("Stage Distribution:", counts)
    print(f"Low-Confidence Flagged: {low_conf_count} ({report['low_conf_ratio']*100:.3f}%)")
    print(f"Results successfully saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
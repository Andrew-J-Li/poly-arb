"""
Event matching: semantic cosine similarity on event TITLE + RULES/DESCRIPTION.

Encodes full event text with sentence-transformers, then
computes cosine similarity for candidate pairs.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from matching.utils import (
    load_kalshi_events,
    load_poly_events,
    get_candidate_pairs,
    save_matches,
)

MODEL_NAME = "all-MiniLM-L6-v2"
THRESHOLD = 0.60  # minimum cosine similarity to keep
MAX_CHARS = 512  # sentence-transformers context limit (in tokens ≈ chars/4)


def truncate(text: str, max_chars: int = MAX_CHARS) -> str:
    """Truncate text to fit within model context window."""
    return text[:max_chars] if len(text) > max_chars else text


def main():
    kalshi = load_kalshi_events()
    poly = load_poly_events()
    pairs = get_candidate_pairs(kalshi, poly)

    print(f"Loading model '{MODEL_NAME}' …")
    model = SentenceTransformer(MODEL_NAME)

    # Encode full texts (truncated for model context)
    print("Encoding Kalshi texts …")
    k_texts = [truncate(t) for t in kalshi["event_text"].tolist()]
    k_embeddings = model.encode(k_texts, show_progress_bar=True, normalize_embeddings=True)

    print("Encoding Polymarket texts …")
    p_texts = [truncate(t) for t in poly["event_text"].tolist()]
    p_embeddings = model.encode(p_texts, show_progress_bar=True, normalize_embeddings=True)

    print(f"Scoring {len(pairs):,} pairs (semantic full) …")
    matches = []
    for ki, pi in pairs:
        score = float(np.dot(k_embeddings[ki], p_embeddings[pi]))

        if score >= THRESHOLD:
            matches.append({
                "kalshi_event": kalshi.at[ki, "event_ticker"],
                "poly_event": poly.at[pi, "event_ticker"],
                "kalshi_title": kalshi.at[ki, "event_title"],
                "poly_title": poly.at[pi, "event_title"],
                "score": round(score, 4),
            })

    df = save_matches(matches, "semantic_full.csv", score_col="score", threshold=0)
    print(f"\nTop 10 matches:")
    print(df.head(10)[["kalshi_title", "poly_title", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
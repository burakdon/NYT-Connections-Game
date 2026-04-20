"""Generate puzzles via Category-First Retrieval (CFR).

Usage:
    # Mode A (0 LLM calls):
    python scripts/cfr/generate_cfr.py --mode remix --count 100

    # Mode B (1 LLM call per puzzle):
    export OPENAI_API_KEY=sk-...
    DRY_RUN=false python scripts/cfr/generate_cfr.py --mode fresh --count 100
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DRY_RUN, LLM_PROVIDER, NYT_PUZZLES_PATH
from src.cfr.embedding_retriever import (
    EmbeddingRetriever,
    load_nyt_word_bank_and_categories,
)
from src.cfr.pipeline import CFRPipeline
from src.cfr.word_bank import build_augmented_word_bank
from src.solvers.roundtable import Roundtable


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    return str(obj)


def main():
    parser = argparse.ArgumentParser(description="Generate puzzles via CFR")
    parser.add_argument(
        "--mode", choices=["remix", "fresh"], default="remix",
        help="remix = 0 LLM calls; fresh = 1 LLM call per puzzle",
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--validate", action="store_true", default=True,
        help="Run roundtable validation (default: yes)",
    )
    parser.add_argument("--no-validate", dest="validate", action="store_false")
    parser.add_argument(
        "--cache",
        type=str,
        default=str(PROJECT_ROOT / "data" / "cache" / "nyt_embeddings_v2.npz"),
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Use only NYT's 4918-word bank (disable WordNet augmentation)",
    )
    args = parser.parse_args()

    print(f"Mode:      {args.mode}")
    print(f"Count:     {args.count}")
    print(f"Validate:  {args.validate}")
    print(f"DRY_RUN:   {DRY_RUN}")
    print()

    # --- Load NYT data (word bank + categories) ---
    print("Loading NYT dataset...")
    nyt_path = NYT_PUZZLES_PATH
    if not Path(nyt_path).exists():
        # Try the "(1)" filename variant
        alt = PROJECT_ROOT / "data" / "nyt_puzzles" / "ConnectionsFinalDataset (1).json"
        if alt.exists():
            nyt_path = str(alt)
        else:
            print(f"ERROR: NYT dataset not found at {nyt_path}")
            return

    nyt_word_bank, nyt_categories = load_nyt_word_bank_and_categories(nyt_path)

    if args.no_augment:
        word_bank = nyt_word_bank
        print(f"  Word bank: {len(word_bank)} NYT-only words (augmentation disabled)")
    else:
        word_bank, composition = build_augmented_word_bank(nyt_word_bank)
        print(
            f"  Word bank: {composition['total']} words "
            f"(NYT: {composition['nyt']}, "
            f"WordNet: +{composition['wordnet_added']}, "
            f"Google 10k: +{composition['google_added']})"
        )

    print(f"  Categories: {len(nyt_categories)} unique NYT categories")

    # --- Load MPNET model ---
    print("Loading MPNET model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2")

    # --- Build retriever ---
    retriever = EmbeddingRetriever(
        word_bank=word_bank,
        nyt_categories=nyt_categories,
        model=model,
        cache_path=args.cache,
    )
    retriever.precompute()

    # --- Build pipeline ---
    llm = None
    if args.mode == "fresh":
        from src.llm_client import LLMClient
        llm = LLMClient()
        print(f"LLM provider: {LLM_PROVIDER}")

    pipeline = CFRPipeline(
        retriever=retriever,
        llm=llm,
        embedding_model=model,
    )
    roundtable = Roundtable(embedding_model=model) if args.validate else None

    # --- Generate ---
    results = {"valid": [], "invalid": [], "errors": []}
    start = time.time()
    nyt_word_set = set(w.upper() for w in nyt_word_bank)

    for i in range(args.count):
        t0 = time.time()
        print(f"\n--- Puzzle {i+1}/{args.count} ---")

        try:
            puzzle = pipeline.generate(mode=args.mode)
            elapsed = time.time() - t0
            print(f"Generated in {elapsed:.2f}s: {puzzle['id']}")

            for g in puzzle["groups"]:
                print(
                    f"  [{g['color']:>6}] {g['category']}: "
                    f"{', '.join(g['words'])} (sim={g['similarity_score']:.3f})"
                )

            if roundtable:
                val = roundtable.validate(puzzle)
                puzzle["metadata"]["solver_agreement"] = val["solver_agreement"]
                puzzle["metadata"]["solvers_used"] = val["solvers_used"]
                puzzle["metadata"]["solver_results"] = val["solver_results"]

                emb_c = val["groups_correct"].get("embedding", 0)
                clust_c = val["groups_correct"].get("clustering", 0)
                accept = (emb_c == 4) or (clust_c == 4)
                puzzle["metadata"]["accepted"] = accept

                print(
                    f"  Validation: {'VALID' if accept else 'INVALID'} "
                    f"(emb={emb_c}/4, clust={clust_c}/4, "
                    f"agree={val['solver_agreement']})"
                )

                if accept:
                    results["valid"].append(puzzle)
                else:
                    results["invalid"].append(puzzle)
            else:
                results["valid"].append(puzzle)

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results["errors"].append({"index": i, "error": str(e)})

    total_time = time.time() - start

    # --- Save ---
    default_dir = PROJECT_ROOT / "data" / "generated" / "cfr" / args.mode
    out_dir = Path(args.output_dir) if args.output_dir else default_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if results["valid"]:
        valid_path = out_dir / f"cfr_{args.mode}_{timestamp}_valid.json"
        with open(valid_path, "w") as f:
            json.dump(results["valid"], f, indent=2, default=_json_default)
        print(f"\nSaved {len(results['valid'])} valid puzzles -> {valid_path}")

    if results["invalid"]:
        invalid_path = out_dir / f"cfr_{args.mode}_{timestamp}_invalid.json"
        with open(invalid_path, "w") as f:
            json.dump(results["invalid"], f, indent=2, default=_json_default)
        print(f"Saved {len(results['invalid'])} invalid puzzles -> {invalid_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"CFR GENERATION SUMMARY ({args.mode})")
    print(f"{'='*60}")
    total = len(results["valid"]) + len(results["invalid"])
    print(f"Total time:         {total_time:.1f}s "
          f"({total_time/max(args.count,1):.2f}s per puzzle)")
    print(f"Generated:          {total}")
    print(f"Valid:              {len(results['valid'])}")
    print(f"Invalid:            {len(results['invalid'])}")
    print(f"Errors:             {len(results['errors'])}")
    if total > 0:
        pass_rate = len(results["valid"]) / total
        print(f"Pass rate:          {pass_rate:.0%}")

    # Rubric-critical metrics
    all_puzzles = results["valid"] + results["invalid"]
    exact_matches = sum(
        1 for p in all_puzzles
        if p.get("metadata", {}).get("exact_group_match", False)
    )
    print(f"Exact NYT group matches (hard-fail check): {exact_matches}")

    # Non-NYT word coverage
    total_words = 0
    non_nyt_words = 0
    for p in all_puzzles:
        for g in p.get("groups", []):
            for w in g.get("words", []):
                total_words += 1
                if w.upper() not in nyt_word_set:
                    non_nyt_words += 1
    if total_words:
        pct = 100.0 * non_nyt_words / total_words
        print(f"Non-NYT words used: {non_nyt_words}/{total_words} ({pct:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

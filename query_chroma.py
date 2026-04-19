import argparse

from foodrag.storage import get_client, get_collection
from foodrag.factcheck import run_fact_check
from foodrag import config


def main():
    parser = argparse.ArgumentParser(description="Query existing Chroma index for top-5 evidence and fact-check.")
    parser.add_argument("claim", help="Claim to evaluate.")
    parser.add_argument("--k", type=int, default=config.TOP_K, help="Top-k chunks.")
    args = parser.parse_args()

    col = get_collection(get_client())
    res = col.query(query_texts=[args.claim], n_results=args.k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]
    evidence = []
    for doc, meta, cid in zip(docs, metas, ids):
        evidence.append({"id": cid, "text": doc, "metadata": meta})

    verdict, reasoning, supporting = run_fact_check(args.claim, evidence)
    print(f"Verdict: {verdict}")
    print(f"Reasoning: {reasoning}")
    print("Evidence:")
    for i, ev in enumerate(evidence, 1):
        mark = "*" if i in supporting else " "
        meta = ev.get("metadata", {})
        print(f"{mark}[{i}] {ev['text']} (file: {meta.get('file')}, tags: {meta.get('tags')})")


if __name__ == "__main__":
    main()

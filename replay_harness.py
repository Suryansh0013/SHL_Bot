"""
Replay harness: scripts each trace's recorded user utterances against your
running /chat API, in order, and compares your agent's FINAL recommendations
turn against the trace's expected final shortlist (by URL, which is the
stable identifier -- names can have minor formatting differences).

This is an approximation of the real evaluator (which uses an LLM to play
the user dynamically). It's most useful for fast local iteration; don't
treat a perfect score here as a guarantee of a perfect score on the real
holdout-trace evaluation.

Usage:
    uvicorn app.main:app --port 8000 &
    python tests/replay_harness.py --url http://localhost:8000
"""
import argparse
import json
import requests

from parse_traces import load_all_traces


def recall_at_k(expected_urls, got_urls, k=10):
    if not expected_urls:
        return None
    top_k = set(got_urls[:k])
    hit = sum(1 for u in expected_urls if u in top_k)
    return hit / len(expected_urls)


def run_trace(trace, api_url, verbose=False):
    messages = []
    last_response = None

    for utterance in trace["user_utterances"]:
        messages.append({"role": "user", "content": utterance})
        resp = requests.post(f"{api_url}/chat", json={"messages": messages}, timeout=30)
        if resp.status_code != 200:
            return {
                "trace_id": trace["trace_id"],
                "error": f"HTTP {resp.status_code}: {resp.text[:300]}",
                "recall": 0.0,
            }
        data = resp.json()
        last_response = data
        messages.append({"role": "assistant", "content": data["reply"]})
        if verbose:
            print(f"  [{trace['trace_id']}] User: {utterance[:70]}")
            print(f"  [{trace['trace_id']}] Agent: {data['reply'][:100]}")
            if data.get("recommendations"):
                names = [r["name"] for r in data["recommendations"]]
                print(f"  [{trace['trace_id']}] -> {names}")

    expected_urls = [r["url"] for r in trace["final_shortlist"]]
    got_urls = [r["url"] for r in (last_response or {}).get("recommendations", [])]
    recall = recall_at_k(expected_urls, got_urls, k=10)

    return {
        "trace_id": trace["trace_id"],
        "expected_count": len(expected_urls),
        "got_count": len(got_urls),
        "recall": recall,
        "expected_names": [r["name"] for r in trace["final_shortlist"]],
        "got_names": [r["name"] for r in (last_response or {}).get("recommendations", [])],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--traces-dir", default="data/traces")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    traces = load_all_traces(args.traces_dir)
    results = []
    for trace in traces:
        print(f"Running {trace['trace_id']}...")
        result = run_trace(trace, args.url, verbose=args.verbose)
        results.append(result)
        recall_str = f"{result['recall']:.2f}" if result.get("recall") is not None else "N/A"
        print(f"  Recall@10: {recall_str}  (expected {result.get('expected_count')}, got {result.get('got_count')})")
        if result.get("error"):
            print(f"  ERROR: {result['error']}")
        print()

    valid = [r["recall"] for r in results if r.get("recall") is not None]
    mean_recall = sum(valid) / len(valid) if valid else 0.0

    print("=" * 50)
    print(f"Mean Recall@10 across {len(valid)} traces: {mean_recall:.3f}")
    print("=" * 50)

    with open("replay_results.json", "w") as f:
        json.dump({"mean_recall_at_10": mean_recall, "results": results}, f, indent=2)
    print("\nFull results written to replay_results.json")


if __name__ == "__main__":
    main()

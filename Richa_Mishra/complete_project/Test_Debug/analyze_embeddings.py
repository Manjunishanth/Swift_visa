import json
from collections import Counter

# Load metadata safely (handles both dict and list)
def load_metadata(path="visa_metadata.json"):
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # If it's a list, convert it into a dict with string keys
    if isinstance(meta, list):
        meta = {str(i): meta[i] for i in range(len(meta))}

    return meta


def print_summary(meta: dict):
    print("\nTotal embeddings:", len(meta))

    # Count per-source
    counter = Counter()
    for v in meta.values():
        src = v.get("source", "UNKNOWN")
        counter[src] += 1

    print("\nPer-file counts:")
    for src, cnt in counter.items():
        print(f" - {src}: {cnt}")

    print("\nUnique source files represented:", len(counter))


def show_issues(meta: dict):
    print("\nChecking for issues...\n")

    # Missing source
    for k, v in meta.items():
        if "source" not in v:
            print("Missing 'source' in entry ID:", k)

    # Missing chunk_id
    for k, v in meta.items():
        if "chunk_id" not in v:
            print("Missing 'chunk_id' in entry ID:", k)


# MAIN
if __name__ == "__main__":
    meta = load_metadata()

    print(f"Loaded {len(meta)} metadata entries.")

    print_summary(meta)
    show_issues(meta)

import argparse
import json
import sys

from .config import VAULT_PATH, CHROMA_PATH, MAX_RESULTS
from .retriever import MemoryRetriever


def main():
    parser = argparse.ArgumentParser(
        prog="aimemory",
        description="Permanent AI memory - query, index, and add markdown memories",
    )
    sub = parser.add_subparsers(dest="cmd")

    q = sub.add_parser("query", help="Search memories semantically")
    q.add_argument("query", help="Search query text")
    q.add_argument(
        "-n", type=int, default=MAX_RESULTS, help="Max results (default: 5)"
    )
    q.add_argument(
        "-c",
        "--category",
        choices=["people", "projects", "concepts", "facts"],
        help="Filter by category",
    )
    q.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )

    sub.add_parser("index", help="Re-index all markdown files in the vault")

    a = sub.add_parser("add", help="Add a new memory note")
    a.add_argument("title", help="Memory title")
    a.add_argument("content", help="Memory content (markdown)")
    a.add_argument(
        "-c",
        "--category",
        default="facts",
        choices=["people", "projects", "concepts", "facts"],
    )
    a.add_argument("-t", "--tags", nargs="*", default=[])

    fg = sub.add_parser("forget", help="Soft-delete a note (moves to .trash/, removes from index)")
    fg.add_argument("slug", help="Note slug (filename stem, e.g. old-preference)")

    rd = sub.add_parser("read", help="Read a note body by slug")
    rd.add_argument("slug", help="Note slug (filename stem, e.g. status-yank)")

    args = parser.parse_args()

    if not args.cmd:
        parser.print_help()
        return

    r = MemoryRetriever(VAULT_PATH, CHROMA_PATH)

    if args.cmd == "query":
        results = r.query(
            args.query, n=args.n, category=args.category, min_relevance=0.0
        )
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for res in results:
                print(
                    f"## {res['title']}  "
                    f"({res['relevance']:.0%} match | {res['category']})"
                )
                if res["tags"]:
                    print(f"tags: {', '.join(res['tags'])}")
                print()
                print(res["content"])
                print("\n---\n")

    elif args.cmd == "index":
        import urllib.request
        import json as _json

        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8420/index", data=b"{}", timeout=30
            ) as resp:
                n = _json.loads(resp.read()).get("chunk_count", 0)
            print(f"Indexed {n} chunks via server")
        except Exception:
            n = r.index()
            print(f"Indexed {n} chunks from vault (server down)")

    elif args.cmd == "add":
        path = r.add_memory(args.title, args.content, args.category, args.tags)
        overlaps = r.check_overlaps(args.title, args.content)
        print(f"Memory added: {path}")
        if overlaps:
            for o in overlaps:
                print(f"  Overlap: {o['slug']} ({o['similarity']:.2f}) — merge + forget {o['slug']}?")

    elif args.cmd == "forget":
        ok = r.delete_note(args.slug)
        if ok:
            print(f"Forgot '{args.slug}' — moved to .trash/ and removed from index.")
        else:
            print(f"Note not found: {args.slug}", file=sys.stderr)
            sys.exit(1)

    elif args.cmd == "read":
        body = r.get_note(args.slug)
        if body is None:
            print(f"No note found for slug: {args.slug}", file=sys.stderr)
            sys.exit(1)
        print(body)


if __name__ == "__main__":
    main()

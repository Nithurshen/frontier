import os
import sys

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

DATA = "data"
CATS = {"cs.CL", "cs.LG", "cs.AI", "cs.IR"}
START = "2023-01-01"


def pull(repo, prefix):
	files = [f for f in list_repo_files(repo, repo_type="dataset")
		 if f.startswith(prefix) and f.endswith(".parquet")]
	out = []
	for f in sorted(files):
		print("fetch", repo, f, flush=True)
		out.append(hf_hub_download(repo, f, repo_type="dataset"))
	return out


def litsearch():
	repo = "princeton-nlp/LitSearch"
	q = pd.concat(pd.read_parquet(f) for f in pull(repo, "query"))
	cols = ["corpusid", "title", "abstract", "citations"]
	c = pd.concat(pd.read_parquet(f, columns=cols)
		      for f in pull(repo, "corpus_clean"))
	q.to_parquet(os.path.join(DATA, "lit_queries.parquet"))
	c.to_parquet(os.path.join(DATA, "lit_corpus.parquet"))
	print("litsearch", len(q), "queries", len(c), "papers")


def keep(cats):
	return any(c in CATS for c in str(cats).split())


def first_date(versions):
	try:
		day = pd.to_datetime(versions[0]["created"], utc=True)
		return day.strftime("%Y-%m-%d")
	except (TypeError, IndexError, KeyError, ValueError):
		return None


def arxiv():
	repo = "librarian-bots/arxiv-metadata-snapshot"
	cols = ["id", "title", "abstract", "categories", "versions",
		"authors"]
	parts = []
	for f in pull(repo, "data/"):
		df = pd.read_parquet(f, columns=cols)
		df = df[df["categories"].map(keep)]
		df["date"] = df["versions"].map(first_date)
		df = df[df["date"].notna() & (df["date"] >= START)]
		parts.append(df.drop(columns=["versions"]))
		print("  kept", len(df), flush=True)
	df = pd.concat(parts).drop_duplicates("id")
	for c in ("title", "abstract", "authors"):
		df[c] = df[c].fillna("").str.split().str.join(" ")
	df = df.sort_values("date").reset_index(drop=True)
	df.to_parquet(os.path.join(DATA, "arxiv.parquet"))
	print("arxiv", len(df), "papers", df["date"].min(), df["date"].max())


def main():
	os.makedirs(DATA, exist_ok=True)
	what = sys.argv[1:] or ["litsearch", "arxiv"]
	if "litsearch" in what:
		litsearch()
	if "arxiv" in what:
		arxiv()


if __name__ == "__main__":
	main()

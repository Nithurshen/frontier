import json
import os
import sys

import pandas as pd

from . import dense, index

DATA = "data"


def corpus(name):
	if name == "lit":
		df = pd.read_parquet(os.path.join(DATA, "lit_corpus.parquet"))
		df["title"] = df["title"].fillna("")
		df["abstract"] = df["abstract"].fillna("")
		return df.reset_index(drop=True)
	return pd.read_parquet(os.path.join(DATA, "arxiv.parquet"))


def main():
	args = sys.argv[1:]
	names = [a for a in args if a in ("lit", "arxiv")] or ["lit", "arxiv"]
	models = [a for a in args if a in dense.MODELS] or ["bge", "e5"]
	for name in names:
		df = corpus(name)
		path = os.path.join(DATA, name, "index.pkl")
		if "dense-only" not in args:
			ix = index.build(name, df["title"], df["abstract"])
			out = os.path.join(DATA, name, "stats.json")
			with open(out, "w") as f:
				json.dump(index.stats(ix), f, indent=1)
		elif not os.path.exists(path):
			sys.exit("build the index first")
		if "index-only" in args:
			continue
		for m in models:
			dense.build(name, m, df["title"], df["abstract"])


if __name__ == "__main__":
	main()

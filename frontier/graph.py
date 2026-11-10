import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA = "data"


def citation_graph(df):
	pos = {int(c): i for i, c in enumerate(df["corpusid"])}
	src, dst = [], []
	for i, cites in enumerate(df["citations"]):
		if cites is None:
			continue
		for c in cites:
			j = pos.get(int(c))
			if j is not None and j != i:
				src.append(i)
				dst.append(j)
	n = len(df)
	data = np.ones(len(src), np.float32)
	return sp.csr_matrix((data, (src, dst)), shape=(n, n))


def pagerank(g, damp=0.85, tol=1e-10, steps=100):
	n = g.shape[0]
	out = np.asarray(g.sum(axis=1)).ravel()
	inv = np.where(out > 0, 1 / np.maximum(out, 1), 0)
	walk = sp.diags(inv) @ g
	p = np.full(n, 1 / n)
	for k in range(steps):
		dangling = p[out == 0].sum()
		nxt = damp * (walk.T @ p + dangling / n) + (1 - damp) / n
		if np.abs(nxt - p).sum() < tol:
			p = nxt
			break
		p = nxt
	return p, k + 1


def build():
	df = pd.read_parquet(os.path.join(DATA, "lit_corpus.parquet"))
	df = df.reset_index(drop=True)
	g = citation_graph(df)
	p, steps = pagerank(g)
	prior = np.log(p * len(p))
	prior = (prior - prior.min()) / (prior.max() - prior.min())
	np.save(os.path.join(DATA, "lit", "pagerank.npy"),
		prior.astype(np.float32))
	indeg = np.asarray(g.sum(axis=0)).ravel()
	print("edges", g.nnz, "steps", steps, "cited docs",
	      int((indeg > 0).sum()))
	return p


if __name__ == "__main__":
	build()

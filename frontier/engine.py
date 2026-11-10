import os

import numpy as np
import pandas as pd

from . import dense, rank, terms
from .index import Index

DATA = "data"
LEX = ("tfidf", "bm25", "ql", "bm25+prf", "bm25+E")
DENSE = ("bge", "e5", "bge+E", "e5+E")
CFG = {
	"alpha": 0.4,
	"alpha_e": 0.55,
	"beta": 0.3,
	"link": 0.3,
	"prior": 0.05,
	"dense": "bge",
	"k1": 0.9,
	"b": 0.4,
}


class Engine:
	def __init__(self, corpus, models=("bge", "e5")):
		self.corpus = corpus
		root = os.path.join(DATA, corpus)
		self.ix = Index.load(os.path.join(root, "index.pkl"))
		self.meta = self.load_meta()
		self.dense = {}
		for m in models:
			p = dense.path_of(corpus, m)
			if os.path.exists(p):
				self.dense[m] = dense.Dense(m, p)
		self.gloss = {}
		self.months = []
		p = os.path.join(DATA, "arxiv", "terms.pkl")
		if os.path.exists(p):
			t = terms.load()
			self.gloss = t["terms"]
			self.months = t["months"]
			self.totals = t["totals"]
		self.prior = None
		p = os.path.join(root, "pagerank.npy")
		if os.path.exists(p):
			self.prior = np.load(p)
		self.cfg = dict(CFG)
		self.qcache = {}

	def load_meta(self):
		if self.corpus == "lit":
			df = pd.read_parquet(os.path.join(DATA,
							  "lit_corpus.parquet"))
			df = df.reset_index(drop=True)
			df["id"] = df["corpusid"].astype(str)
			df["date"] = ""
			return df[["id", "title", "abstract", "date"]]
		df = pd.read_parquet(os.path.join(DATA, "arxiv.parquet"))
		return df[["id", "title", "abstract", "date", "categories",
			   "authors"]]

	def spot(self, text):
		return terms.spot(text, self.gloss) if self.gloss else []

	def expand_terms(self, found):
		out = {}
		for t in found:
			for w, x in self.gloss[t]["expand"].items():
				out[w] = max(out.get(w, 0.0), x)
		return out

	def expand_text(self, text, found):
		notes = [f"{self.gloss[t]['surface']}: "
			 f"{self.gloss[t]['origin_title']}" for t in found]
		if not notes:
			return text
		return text + " (" + "; ".join(notes) + ")"

	def links(self, found):
		if self.corpus != "arxiv":
			return []
		return [self.gloss[t]["origin"] for t in found
			if self.gloss[t].get("named")]

	def lexical(self, text, model, found=()):
		kb = {"k1": self.cfg["k1"], "b": self.cfg["b"]}
		if model == "bm25+prf":
			return rank.lexical(self.ix, text, "bm25", prf=True,
					    **kb)
		if model == "bm25+E":
			extra = self.expand_terms(found)
			return rank.lexical(self.ix, text, "bm25", extra=extra,
					    beta=self.cfg["beta"], **kb)
		if model == "bm25":
			return rank.lexical(self.ix, text, model, **kb)
		return rank.lexical(self.ix, text, model)

	def semantic(self, text, model, found=()):
		base = model.split("+")[0]
		if model.endswith("+E"):
			text = self.expand_text(text, found)
		key = (base, text)
		if key not in self.qcache:
			self.qcache[key] = dense.encode_queries(base, [text])[0]
		return self.dense[base].scores(self.qcache[key])

	def scores(self, text, system, found):
		if system in LEX:
			return self.lexical(text, system, found)
		if system in DENSE:
			return self.semantic(text, system, found)
		raise KeyError(system)

	def mask(self, s, before):
		if before:
			dates = self.meta["date"].to_numpy()
			s = s.copy()
			s[: len(dates)][dates >= before] = -1e9
		return s

	def parts(self, text, before=None):
		found = self.spot(text)
		d = self.cfg["dense"]
		p = {"found": found}
		p["bm25"] = self.mask(self.lexical(text, "bm25"), before)
		p["bm25+E"] = self.mask(self.lexical(text, "bm25+E", found),
					before)
		p["dense"] = self.mask(self.semantic(text, d), before)
		p["dense+E"] = self.mask(self.semantic(text, d + "+E", found),
					 before)
		link = self.links(found)
		if before and link:
			dates = self.meta["date"].to_numpy()
			link = [i for i in link if dates[i] < before]
		p["link"] = link
		return p

	def combine(self, p, system, cfg=None, k=rank.K):
		cfg = cfg or self.cfg
		use = system in ("hybrid+E", "frontier", "frontier+pr")
		lex = p["bm25+E"] if use else p["bm25"]
		sem = p["dense+E"] if use else p["dense"]
		a = cfg["alpha_e"] if use and p["found"] else cfg["alpha"]
		link = p["link"] if system.startswith("frontier") else []
		pool, acc = rank.fuse([lex, sem], [a, 1 - a],
				       extra=link or None)
		if link:
			acc[np.isin(pool, link)] += cfg["link"]
		if system.endswith("+pr") and self.prior is not None:
			acc += cfg["prior"] * self.prior[pool]
		return rank.order(pool, acc, k)

	def search(self, text, system="frontier", k=rank.K, before=None):
		if system in LEX or system in DENSE:
			found = self.spot(text)
			s = self.mask(self.scores(text, system, found), before)
			return rank.top(s, k), found
		p = self.parts(text, before)
		return self.combine(p, system, k=k), p["found"]

	def ingest(self, df):
		if df is None or not len(df):
			return 0
		df = df[~df["id"].isin(set(self.meta["id"]))]
		if not len(df):
			return 0
		ti, ab = list(df["title"]), list(df["abstract"])
		self.ix.add(ti, ab)
		for d in self.dense.values():
			d.add(ti, ab)
		cols = list(self.meta.columns)
		self.meta = pd.concat([self.meta, df[cols]], ignore_index=True)
		return len(df)

	def doc(self, i):
		r = self.meta.iloc[i]
		return {"i": int(i), "id": r["id"], "title": r["title"],
			"abstract": r["abstract"], "date": r["date"]}

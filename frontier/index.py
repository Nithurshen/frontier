import os
import pickle
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp

from .text import tokens

TITLE_W = 2


def doc_terms(title, abstract):
	c = Counter(tokens(abstract))
	for t in tokens(title):
		c[t] += TITLE_W
	return c


def collect(job):
	titles, abstracts = job
	vocab = {}
	rows, cols, vals = [], [], []
	for i, (ti, ab) in enumerate(zip(titles, abstracts)):
		for t, f in doc_terms(ti, ab).items():
			k = vocab.setdefault(t, len(vocab))
			rows.append(k)
			cols.append(i)
			vals.append(f)
	return (list(vocab), np.array(rows, np.int32),
		np.array(cols, np.int32), np.array(vals, np.float32))


def chunks(titles, abstracts, size):
	for i in range(0, len(titles), size):
		yield titles[i:i + size], abstracts[i:i + size]


class Index:
	def __init__(self):
		self.terms = []
		self.vocab = {}
		self.post = sp.csr_matrix((0, 0), dtype=np.float32)
		self.dl = np.zeros(0, np.float32)
		self.aux = None
		self.aux_dl = np.zeros(0, np.float32)
		self.merges = 0

	@property
	def n(self):
		return len(self.dl) + len(self.aux_dl)

	def tid(self, t):
		return self.vocab.get(t, -1)

	def add_terms(self, terms):
		for t in terms:
			if t not in self.vocab:
				self.vocab[t] = len(self.terms)
				self.terms.append(t)

	def part(self, titles, abstracts, workers):
		size = max(1, len(titles) // (workers * 4) + 1)
		jobs = list(chunks(titles, abstracts, size))
		if workers > 1 and len(jobs) > 1:
			with Pool(workers) as pool:
				parts = pool.map(collect, jobs)
		else:
			parts = [collect(j) for j in jobs]
		rows, cols, vals = [], [], []
		base = 0
		for (vocab, r, c, v), job in zip(parts, jobs):
			self.add_terms(vocab)
			remap = [self.vocab[t] for t in vocab]
			remap = np.array(remap, np.int32)
			rows.append(remap[r] if len(r) else r)
			cols.append(c + base)
			vals.append(v)
			base += len(job[0])
		shape = (len(self.terms), len(titles))
		ij = (np.concatenate(rows), np.concatenate(cols))
		m = sp.csr_matrix((np.concatenate(vals), ij), shape=shape,
				  dtype=np.float32)
		m.sum_duplicates()
		return m

	def build(self, titles, abstracts, workers=12):
		self.post = self.part(titles, abstracts, workers)
		self.dl = np.asarray(self.post.sum(axis=0)).ravel()
		self.finish()

	def pad(self, m):
		if m.shape[0] < len(self.terms):
			m = m.copy()
			m.resize((len(self.terms), m.shape[1]))
		return m

	def add(self, titles, abstracts, limit=5000):
		m = self.part(titles, abstracts, 1)
		if self.aux is None:
			self.aux = m
		else:
			self.aux = sp.hstack([self.pad(self.aux), m]).tocsr()
		self.aux_dl = np.asarray(self.aux.sum(axis=0)).ravel()
		self.post = self.pad(self.post)
		self.aux = self.pad(self.aux)
		if self.aux.shape[1] >= limit:
			self.merge()
		self.finish()

	def merge(self):
		if self.aux is None:
			return
		self.post = sp.hstack([self.pad(self.post),
				       self.pad(self.aux)]).tocsr()
		self.dl = np.concatenate([self.dl, self.aux_dl])
		self.aux = None
		self.aux_dl = np.zeros(0, np.float32)
		self.merges += 1

	def matrices(self):
		out = [(self.post, 0)]
		if self.aux is not None:
			out.append((self.aux, len(self.dl)))
		return out

	def finish(self):
		self.df = np.zeros(len(self.terms), np.float32)
		self.cf = np.zeros(len(self.terms), np.float32)
		for m, _ in self.matrices():
			self.df += np.diff(m.indptr)
			self.cf += np.asarray(m.sum(axis=1)).ravel()
		self.all_dl = np.concatenate([self.dl, self.aux_dl])
		self.avgdl = float(self.all_dl.mean()) if self.n else 0.0
		self.total = float(self.cf.sum())
		self.idf = np.log(self.n / np.maximum(self.df, 1)).astype(
			np.float32)
		self.fwd = None
		self.norm = None

	def forward(self):
		if self.fwd is None:
			parts = [m for m, _ in self.matrices()]
			self.fwd = sp.hstack(parts).T.tocsr()
		return self.fwd

	def lnc_norm(self):
		if self.norm is None:
			f = self.forward().copy()
			f.data = (1 + np.log(f.data)) ** 2
			self.norm = np.sqrt(np.asarray(f.sum(axis=1)).ravel())
		return self.norm

	def postings(self, t):
		for m, base in self.matrices():
			if t >= m.shape[0]:
				continue
			a, b = m.indptr[t], m.indptr[t + 1]
			yield m.indices[a:b] + base, m.data[a:b]

	def save(self, path):
		fwd, norm = self.fwd, self.norm
		self.fwd = self.norm = None
		with open(path, "wb") as f:
			pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
		self.fwd, self.norm = fwd, norm

	@classmethod
	def load(cls, path):
		with open(path, "rb") as f:
			return pickle.load(f)


def vbyte_size(gaps):
	return int(np.sum(np.floor(np.log2(np.maximum(gaps, 1)) / 7) + 1))


def gamma_bits(gaps):
	return int(np.sum(2 * np.floor(np.log2(np.maximum(gaps, 1))) + 1))


def stats(ix):
	m = ix.post
	n_post = int(m.nnz)
	gaps = []
	for t in range(m.shape[0]):
		a, b = m.indptr[t], m.indptr[t + 1]
		if b > a:
			d = m.indices[a:b]
			gaps.append(np.diff(d, prepend=-1).astype(np.int64))
	gaps = np.concatenate(gaps)
	cf = np.sort(ix.cf)[::-1]
	cf = cf[cf > 0]
	ranks = np.arange(1, len(cf) + 1)
	top = slice(10, min(len(cf), 10000))
	slope = np.polyfit(np.log(ranks[top]), np.log(cf[top]), 1)[0]
	return {
		"docs": int(ix.n),
		"terms": int(len(ix.terms)),
		"postings": n_post,
		"tokens": int(ix.total),
		"avgdl": round(ix.avgdl, 2),
		"raw_mb": round(n_post * 4 / 2 ** 20, 2),
		"vbyte_mb": round(vbyte_size(gaps) / 2 ** 20, 2),
		"gamma_mb": round(gamma_bits(gaps) / 8 / 2 ** 20, 2),
		"zipf_slope": round(float(slope), 3),
		"merges": ix.merges,
	}


def build(name, titles, abstracts, root="data"):
	t0 = time.time()
	ix = Index()
	ix.build(list(titles), list(abstracts))
	path = os.path.join(root, name, "index.pkl")
	os.makedirs(os.path.dirname(path), exist_ok=True)
	ix.save(path)
	print(name, "index", ix.n, "docs", len(ix.terms), "terms",
	      round(time.time() - t0, 1), "s", flush=True)
	return ix

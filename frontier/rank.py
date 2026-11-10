from collections import Counter

import numpy as np

from .text import tokens

K = 1000


def query_terms(ix, text):
	q = Counter()
	for t, f in Counter(tokens(text)).items():
		k = ix.tid(t)
		if k >= 0:
			q[k] += f
	return q


def bm25(ix, q, k1=0.9, b=0.4):
	s = np.zeros(ix.n, np.float32)
	norm = k1 * (1 - b + b * ix.all_dl / ix.avgdl)
	for t, w in q.items():
		df = ix.df[t]
		idf = np.log(1 + (ix.n - df + 0.5) / (df + 0.5))
		for d, tf in ix.postings(t):
			s[d] += w * idf * tf * (k1 + 1) / (tf + norm[d])
	return s


def tfidf(ix, q):
	s = np.zeros(ix.n, np.float32)
	for t, f in q.items():
		w = (1 + np.log(f)) * ix.idf[t]
		for d, tf in ix.postings(t):
			s[d] += w * (1 + np.log(tf))
	return s / np.maximum(ix.lnc_norm(), 1e-9)


def ql(ix, q, mu=1000):
	s = np.zeros(ix.n, np.float32)
	hit = np.zeros(ix.n, bool)
	for t, f in q.items():
		p = ix.cf[t] / ix.total
		for d, tf in ix.postings(t):
			s[d] += f * np.log(1 + tf / (mu * p))
			hit[d] = True
	s += sum(q.values()) * np.log(mu / (ix.all_dl + mu))
	s[~hit] = -1e9
	return s


def top(s, k=K):
	k = min(k, len(s))
	i = np.argpartition(-s, k - 1)[:k]
	return i[np.argsort(-s[i], kind="stable")]


def feedback(ix, q, s, docs=10, n_terms=20, beta=0.5):
	fwd = ix.forward()
	acc = Counter()
	for d in top(s, docs):
		a, b = fwd.indptr[d], fwd.indptr[d + 1]
		tf = fwd.data[a:b]
		w = tf / tf.sum() * ix.idf[fwd.indices[a:b]]
		for t, x in zip(fwd.indices[a:b], w):
			acc[t] += x / docs
	best = acc.most_common(n_terms)
	if not best:
		return q
	hi = best[0][1]
	out = Counter({t: float(f) for t, f in q.items()})
	for t, x in best:
		out[t] += beta * x / hi
	return out


def lexical(ix, text, model="bm25", prf=False, extra=None, beta=0.3,
	    k1=0.9, b=0.4):
	q = query_terms(ix, text)
	if extra:
		hi = max(extra.values())
		for t, x in extra.items():
			k = ix.tid(t)
			if k >= 0:
				q[k] += beta * x / hi
	if not q:
		return np.zeros(ix.n, np.float32)
	fn = {"bm25": bm25, "tfidf": tfidf, "ql": ql}[model]
	args = (k1, b) if model == "bm25" else ()
	s = fn(ix, q, *args)
	if prf:
		s = fn(ix, feedback(ix, q, s), *args)
	return s


def minmax(s, ids):
	v = s[ids]
	lo, hi = float(v.min()), float(v.max())
	return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def fuse(scores, weights, k=K, extra=None):
	lists = [top(s, k) for s in scores]
	if extra is not None:
		lists.append(np.asarray(extra, np.int64))
	pool = np.unique(np.concatenate(lists))
	acc = np.zeros(len(pool))
	for s, w in zip(scores, weights):
		acc += w * minmax(s, pool)
	return pool, acc


def order(pool, acc, k=K):
	i = np.argsort(-acc, kind="stable")[:k]
	return pool[i]

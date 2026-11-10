import os
import pickle
import re
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
from wordfreq import zipf_frequency

from .text import grams, sentences, tokens, words

DATA = "data"
START = "2024-07"
UNTIL = "2026-09-01"
MIN_TOTAL = 30
MIN_RECENT = 20
GROWTH = 8
CONTRAST = 0.02
TITLED = 0.05
COMMON = 2.5
MARGIN = 0.065
KEEP = 1500
SIZE = re.compile(r"[-\s]?\d+(\.\d+)?[bm]$")
INTRO = re.compile(r"\b(we|this paper|this work) "
		   r"(propose|introduce|present|release)", re.I)


def month_counts(texts):
	c = Counter()
	for t in texts:
		c.update(grams(t))
	return Counter({k: v for k, v in c.items() if v >= 3})


def count_matrix(df, workers=12):
	months = sorted(df["month"].unique())
	jobs = [list(df.loc[df["month"] == m, "text"]) for m in months]
	with Pool(workers) as pool:
		parts = pool.map(month_counts, jobs)
	vocab = {}
	for p in parts:
		for k in p:
			vocab.setdefault(k, len(vocab))
	mat = np.zeros((len(vocab), len(months)), np.int32)
	for j, p in enumerate(parts):
		ids = np.fromiter((vocab[k] for k in p), np.int64, len(p))
		mat[ids, j] = np.fromiter(p.values(), np.int64, len(p))
	totals = df.groupby("month").size().reindex(months).to_numpy()
	return months, list(vocab), mat, totals


def kleinberg(r, d, s=3.0, gamma=1.0):
	n = len(r)
	p0 = r.sum() / d.sum()
	p1 = min(s * p0, 0.9999)
	c0 = -(r * np.log(p0) + (d - r) * np.log(1 - p0))
	c1 = -(r * np.log(p1) + (d - r) * np.log(1 - p1))
	up = gamma * np.log(n)
	cost = np.array([0.0, np.inf])
	back = np.zeros((n, 2), np.int8)
	for t in range(n):
		a = min(cost[0], cost[1])
		back[t, 0] = 0 if cost[0] <= cost[1] else 1
		b0 = cost[0] + up
		b1 = cost[1]
		back[t, 1] = 0 if b0 < b1 else 1
		cost = np.array([a + c0[t], min(b0, b1) + c1[t]])
	state = np.zeros(n, np.int8)
	state[-1] = 0 if cost[0] <= cost[1] else 1
	for t in range(n - 1, 0, -1):
		state[t - 1] = back[t, state[t]]
	return state, c0 - c1


def last_burst(state, gain):
	idx = np.flatnonzero(state)
	if not len(idx):
		return None
	end = idx[-1]
	start = end
	while start > 0 and state[start - 1]:
		start -= 1
	return start, end, float(gain[start:end + 1].sum())


def background(workers=12):
	df = pd.read_parquet(os.path.join(DATA, "lit_corpus.parquet"))
	texts = list(df["title"].fillna("") + ". " +
		     df["abstract"].fillna(""))
	with Pool(workers) as pool:
		parts = pool.map(month_counts,
				 [texts[i::workers] for i in range(workers)])
	bg = Counter()
	for p in parts:
		bg.update(p)
	return bg, len(texts)


def detect(months, vocab, mat, totals, bg, nbg):
	recent = mat[:, -6:].sum(axis=1)
	rate_r = recent / totals[-6:].sum()
	rate_e = (mat[:, :18].sum(axis=1) + 1) / totals[:18].sum()
	rate_b = np.array([bg.get(v, 0) for v in vocab]) / nbg
	ok = ((mat.sum(axis=1) >= MIN_TOTAL) & (recent >= MIN_RECENT) &
	      (rate_r / rate_e >= GROWTH) & (rate_b <= CONTRAST * rate_r))
	first = months.index(START)
	rows = []
	for i in np.flatnonzero(ok):
		term = vocab[i]
		if re.fullmatch(r"[\d\.\-]+", term) or len(term) < 3:
			continue
		state, gain = kleinberg(mat[i].astype(float),
					totals.astype(float))
		b = last_burst(state, gain)
		if b is None or b[0] < first:
			continue
		start, end, weight = b
		before = mat[i, :start].sum() / max(totals[:start].sum(), 1)
		span = slice(start, end + 1)
		during = mat[i, span].sum() / totals[span].sum()
		if before > 0.25 * during:
			continue
		rows.append((term, months[start], months[end], weight,
			     int(mat[i].sum()), i))
	rows.sort(key=lambda x: -x[3])
	return rows


def termhood(term, form, ids, df):
	if form[1:] != form[1:].lower() or re.search(r"\d", form):
		return True
	if any(w.endswith("ly") for w in term.split()):
		return False
	if " " not in term and "-" not in term:
		if zipf_frequency(term, "en") >= COMMON:
			return False
	titles = sum(has(term, df.at[i, "title"]) for i in ids[:400])
	return titles / min(len(ids), 400) >= TITLED


def dedupe(rows, mat):
	kept = []
	for r in rows:
		words = set(r[0].split())
		clash = None
		for j, k in enumerate(kept):
			other = set(k[0].split())
			if words <= other or other <= words:
				lo, hi = sorted([r[4], k[4]])
				if lo / hi > 0.6:
					clash = j
					break
		if clash is None:
			kept.append(r)
		elif len(words) > len(set(kept[clash][0].split())):
			kept[clash] = r
		if len(kept) >= KEEP:
			break
	return kept


def find_docs(job):
	texts, wanted = job
	out = []
	for i, t in enumerate(texts):
		hit = grams(t) & wanted
		if hit:
			out.append((i, hit))
	return out


def term_docs(df, wanted, workers=12):
	texts = list(df["text"])
	size = len(texts) // (workers * 4) + 1
	jobs = [(texts[i:i + size], wanted)
		for i in range(0, len(texts), size)]
	with Pool(workers) as pool:
		parts = pool.map(find_docs, jobs)
	docs = {w: [] for w in wanted}
	for j, part in enumerate(parts):
		for i, hit in part:
			for w in hit:
				docs[w].append(j * size + i)
	return docs


def surface(term, texts):
	pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) +
			 r"(?![A-Za-z0-9])", re.I)
	c = Counter()
	for t in texts:
		c.update(pat.findall(t))
	return c.most_common(1)[0][0] if c else term


def has(term, text):
	return term in grams(text)


def evidence(term, i, df):
	title = df.at[i, "title"]
	low = " ".join(words(title.lower()))
	rest = low[len(term):]
	if low.startswith(term) and (not rest or rest[0] == " "):
		return "head"
	if has(term, title):
		return "title"
	for x in sentences(df.at[i, "abstract"]):
		if INTRO.search(x) and has(term, x):
			return "intro"
	return None


def origin(term, ids, df, vec, start):
	ids = np.array(ids)
	month = df["month"].to_numpy()[ids]
	burst = ids[month >= start]
	if not len(burst):
		burst = ids
	pick = burst[np.linspace(0, len(burst) - 1,
				 min(len(burst), 400)).astype(int)]
	centre = vec[pick].astype(np.float32).mean(axis=0)
	centre /= np.linalg.norm(centre) + 1e-9
	cands = [(i, evidence(term, i, df)) for i in ids[:300]]
	cands = [(i, how) for i, how in cands if how]
	if not cands:
		return int(ids[0]), "first", 0.0
	sims = vec[[i for i, _ in cands]].astype(np.float32) @ centre
	near = np.flatnonzero(sims >= sims.max() - MARGIN)
	head = [k for k in near if cands[k][1] == "head"]
	k = head[0] if head else near[0]
	return int(cands[k][0]), cands[k][1], float(sims[k])


def family(term, out):
	base = SIZE.sub("", term)
	if base != term and base in out:
		return out[base]
	return None


def named(form):
	return form[1:] != form[1:].lower() or bool(re.search(r"\d", form))


def definition(term, abstract):
	for s in sentences(abstract):
		if has(term, s):
			return s
	return sentences(abstract)[0] if abstract else ""


def expansion(ix, title, abstract, n=10):
	c = Counter(tokens(abstract))
	for t in tokens(title):
		c[t] += 2
	sc = {}
	for t, f in c.items():
		k = ix.tid(t)
		if k >= 0:
			sc[t] = (1 + np.log(f)) * float(ix.idf[k])
	best = sorted(sc.items(), key=lambda x: -x[1])[:n]
	return dict(best)


def build(ix=None):
	t0 = time.time()
	df = pd.read_parquet(os.path.join(DATA, "arxiv.parquet"))
	df = df[df["date"] < UNTIL].copy()
	df["month"] = df["date"].str[:7]
	df["text"] = df["title"] + ". " + df["abstract"]
	months, vocab, mat, totals = count_matrix(df)
	bg, nbg = background()
	print("candidates", len(vocab), round(time.time() - t0), "s",
	      flush=True)
	rows = detect(months, vocab, mat, totals, bg, nbg)
	docs = term_docs(df, {r[0] for r in rows})
	good = []
	for r in rows:
		ids = docs[r[0]]
		if not ids:
			continue
		form = surface(r[0], df["text"].iloc[ids[:30]])
		if termhood(r[0], form, ids, df):
			good.append(r + (form,))
	good = dedupe(good, mat)
	print("bursty", len(rows), "terms", len(good),
	      round(time.time() - t0), "s", flush=True)
	if ix is None:
		from .index import Index
		ix = Index.load(os.path.join(DATA, "arxiv", "index.pkl"))
	vec = np.load(os.path.join(DATA, "arxiv", "dense_bge.npy"),
		      mmap_mode="r")
	out = {}
	for term, start, end, weight, total, i, form in good:
		ids = docs[term]
		o, how, sim = origin(term, ids, df, vec, start)
		ti, ab = df.at[o, "title"], df.at[o, "abstract"]
		out[term] = {
			"term": term,
			"surface": form,
			"start": start,
			"end": end,
			"weight": round(weight, 2),
			"total": total,
			"background": int(bg.get(term, 0)),
			"counts": mat[i].tolist(),
			"origin": int(o),
			"origin_id": df.at[o, "id"],
			"origin_title": ti,
			"origin_date": df.at[o, "date"],
			"how": how,
			"sim": round(sim, 3),
			"definition": definition(term, ab),
			"expand": expansion(ix, ti, ab),
			"docs": ids,
		}
	for term, v in out.items():
		v["named"] = named(v["surface"])
		base = family(term, out)
		if base is not None:
			for k in ("origin", "origin_id", "origin_title",
				  "origin_date", "how", "expand"):
				v[k] = base[k]
	res = {"months": months, "totals": totals.tolist(), "terms": out}
	with open(os.path.join(DATA, "arxiv", "terms.pkl"), "wb") as f:
		pickle.dump(res, f)
	tab = pd.DataFrame([{k: v[k] for k in ("term", "surface", "start",
		"weight", "total", "background", "origin_id", "origin_title")}
		for v in out.values()])
	tab.to_csv(os.path.join(DATA, "arxiv", "terms.csv"), index=False)
	print("saved", len(out), "terms", round(time.time() - t0), "s")
	return res


def load():
	with open(os.path.join(DATA, "arxiv", "terms.pkl"), "rb") as f:
		return pickle.load(f)


def spot(text, glossary):
	found = grams(text) & glossary.keys()
	out = []
	for t in sorted(found, key=lambda x: -len(x)):
		if not any(set(t.split()) <= set(o.split()) for o in out):
			out.append(t)
	return out


if __name__ == "__main__":
	build()

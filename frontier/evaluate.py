import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from . import rank
from .engine import Engine

DATA = "data"
OUT = "results"
DEPTH = 100
SINGLE = ["tfidf", "ql", "bm25", "bm25+prf", "bm25+E", "e5", "e5+E",
	  "bge", "bge+E"]
FUSED = ["hybrid", "hybrid+E", "frontier"]
PAPER = {
	"BM25 (paper)": (38.5, 37.4, 62.6, 48.6, 50.0, 39.9),
	"GTR-T5-large (paper)": (38.5, 45.7, 40.8, 37.1, 39.6, 43.8),
	"Instructor-XL (paper)": (48.9, 56.3, 55.9, 57.1, 52.3, 56.5),
	"E5-large-v2 (paper)": (50.4, 55.8, 62.6, 54.3, 56.2, 55.4),
	"GritLM-7B (paper)": (67.7, 69.7, 82.5, 74.3, 74.8, 70.8),
}
GRID = {
	"alpha": [0.2, 0.3, 0.4, 0.5, 0.6],
	"alpha_e": [0.3, 0.4, 0.5, 0.6, 0.7],
	"link": [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5],
}
BETAS = [0.1, 0.3, 0.5, 1.0]


def lit_queries():
	q = pd.read_parquet(os.path.join(DATA, "lit_queries.parquet"))
	c = pd.read_parquet(os.path.join(DATA, "lit_corpus.parquet"))
	pos = {int(x): i for i, x in enumerate(c["corpusid"])}
	rows = []
	for i, r in enumerate(q.itertuples()):
		rel = {pos[int(x)] for x in r.corpusids if int(x) in pos}
		inline = r.query_set.startswith("inline")
		kind = "inline" if inline else "author"
		rows.append({"qid": f"l{i:03d}", "query": r.query, "rel": rel,
			     "kind": kind, "specific": int(r.specificity) == 1,
			     "before": None})
	return rows


def fresh_queries():
	q = pd.read_parquet(os.path.join(DATA, "fresh_queries.parquet"))
	a = pd.read_parquet(os.path.join(DATA, "arxiv.parquet"),
			    columns=["id", "date"])
	pos = dict(zip(a["id"], range(len(a))))
	date = dict(zip(a["id"], a["date"]))
	papers = sorted(q["citing"].unique())
	split = {p: ("dev" if i % 2 else "test") for i, p in enumerate(papers)}
	rows = []
	for r in q.itertuples():
		rows.append({"qid": r.qid, "query": r.query,
			     "rel": {pos[t] for t in r.targets},
			     "era": r.era, "split": split[r.citing],
			     "before": date[r.citing]})
	return rows


def metrics(ranked, rel):
	ranked = list(ranked[:DEPTH])
	hits = [1 if d in rel else 0 for d in ranked]
	n = len(rel)
	dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits[:10]))
	idcg = sum(1 / np.log2(i + 2) for i in range(min(n, 10)))
	first = next((i for i, h in enumerate(hits) if h), None)
	return {
		"R@5": sum(hits[:5]) / n,
		"R@20": sum(hits[:20]) / n,
		"R@100": sum(hits) / n,
		"P@5": sum(hits[:5]) / 5,
		"nDCG@10": dcg / idcg,
		"MRR": 0.0 if first is None else 1 / (first + 1),
		"AP": ap(hits, n),
	}


def ap(hits, n):
	got, acc = 0, 0.0
	for i, h in enumerate(hits):
		if h:
			got += 1
			acc += got / (i + 1)
	return acc / n


def pr_points(ranked, rel):
	hits = np.array([1 if d in rel else 0 for d in ranked[:DEPTH]])
	if not hits.any():
		return np.zeros(11)
	cum = np.cumsum(hits)
	prec = cum / np.arange(1, len(hits) + 1)
	rec = cum / len(rel)
	out = []
	for lv in np.linspace(0, 1, 11):
		ok = prec[rec >= lv]
		out.append(ok.max() if len(ok) else 0.0)
	return np.array(out)


def run_query(eng, q):
	out = {}
	times = {}
	for s in SINGLE:
		t0 = time.time()
		r, _ = eng.search(q["query"], s, k=DEPTH, before=q["before"])
		times[s] = time.time() - t0
		out[s] = r
	t0 = time.time()
	p = eng.parts(q["query"], q["before"])
	base = time.time() - t0
	for s in FUSED + (["frontier+pr"] if eng.prior is not None else []):
		t0 = time.time()
		out[s] = eng.combine(p, s, k=DEPTH)
		times[s] = base + time.time() - t0
	return out, times, p


def run_set(eng, qs, label):
	rows, pr, lat = [], {}, {}
	t0 = time.time()
	for i, q in enumerate(qs):
		out, times, p = run_query(eng, q)
		for s, ranked in out.items():
			m = metrics(ranked, q["rel"])
			rows.append({"qid": q["qid"], "system": s,
				     "terms": "; ".join(p["found"]), **m})
			pr.setdefault(s, []).append(pr_points(ranked, q["rel"]))
			lat.setdefault(s, []).append(times[s])
		if i % 50 == 0:
			print(f"  {label} {i}/{len(qs)} "
			      f"{time.time() - t0:.0f}s", flush=True)
	df = pd.DataFrame(rows)
	meta = pd.DataFrame(qs).drop(columns=["rel", "query"])
	df = df.merge(meta, on="qid")
	pr = {s: np.mean(v, axis=0).round(4).tolist() for s, v in pr.items()}
	lat = {s: round(1000 * float(np.mean(v)), 1) for s, v in lat.items()}
	return df, pr, lat


def restrict(parts, rel):
	keys = ("bm25", "bm25+E", "dense", "dense+E")
	ids = [rank.top(p[k]) for p in parts.values() for k in keys]
	ids += [np.asarray(p["link"], np.int64) for p in parts.values()]
	pool = np.unique(np.concatenate(ids))
	where = {int(d): i for i, d in enumerate(pool)}
	out = {}
	for b, p in parts.items():
		small = {k: p[k][pool] for k in keys}
		small["found"] = p["found"]
		small["link"] = [where[int(d)] for d in p["link"]]
		out[b] = small
	return out, {where.get(d, -1 - k) for k, d in enumerate(rel)}


def tune_bm25(eng, qs):
	best, score = None, -1
	for k1, b in itertools.product([0.6, 0.9, 1.2, 1.5],
				       [0.3, 0.5, 0.75, 0.9]):
		eng.cfg.update(k1=k1, b=b)
		v = np.mean([metrics(eng.search(q["query"], "bm25", DEPTH,
						q["before"])[0],
				     q["rel"])["nDCG@10"] for q in qs])
		if v > score:
			best, score = (k1, b), v
	eng.cfg.update(k1=best[0], b=best[1])
	print("bm25", best, round(score, 4), flush=True)
	return best


def tune(eng, qs):
	tune_bm25(eng, qs)
	cache = []
	for i, q in enumerate(qs):
		parts = {}
		for b in BETAS:
			eng.cfg["beta"] = b
			parts[b] = eng.parts(q["query"], q["before"])
		cache.append(restrict(parts, q["rel"]))
		if i % 50 == 0:
			print(f"  tune {i}/{len(qs)}", flush=True)
	best, score = None, -1
	trace = []
	for b in BETAS:
		for a, ae, lk in itertools.product(*GRID.values()):
			cfg = dict(eng.cfg, alpha=a, alpha_e=ae, link=lk,
				   beta=b)
			v = np.mean([metrics(eng.combine(p[b], "frontier", cfg,
							 DEPTH), rel)["nDCG@10"]
				     for p, rel in cache])
			trace.append({**cfg, "nDCG@10": float(v)})
			if v > score:
				best, score = cfg, v
	hyb, hs = None, -1
	for a in GRID["alpha"]:
		cfg = dict(eng.cfg, alpha=a)
		v = np.mean([metrics(eng.combine(p[BETAS[0]], "hybrid", cfg,
						 DEPTH), rel)["nDCG@10"]
			     for p, rel in cache])
		if v > hs:
			hyb, hs = a, v
	best["alpha"] = hyb
	print("tuned", best, round(score, 4), "hybrid alpha", hyb, flush=True)
	return best, trace


def table(df, cols, by=None):
	g = ["system"] + ([by] if by else [])
	return (df.groupby(g)[cols].mean() * 100).round(1)


def lit_table(df):
	rows = {}
	for s, d in df.groupby("system"):
		sp = d[d["specific"]]
		br = d[~d["specific"]]
		r = []
		for kind in ("inline", "author"):
			r.append(sp.loc[sp["kind"] == kind, "R@5"].mean())
			r.append(br.loc[br["kind"] == kind, "R@20"].mean())
		r += [sp["R@5"].mean(), br["R@20"].mean(),
		      d["nDCG@10"].mean(), d["MRR"].mean()]
		rows[s] = [round(100 * x, 1) for x in r]
	for s, v in PAPER.items():
		rows[s] = list(v) + [None, None]
	cols = ["inline spec R@5", "inline broad R@20", "author spec R@5",
		"author broad R@20", "avg spec R@5", "avg broad R@20",
		"nDCG@10", "MRR"]
	return pd.DataFrame.from_dict(rows, orient="index", columns=cols)


def significance(df, a, b, metric):
	x = df[df["system"] == a].set_index("qid")[metric]
	y = df[df["system"] == b].set_index("qid")[metric]
	x, y = x.align(y, join="inner")
	t = stats.ttest_rel(x, y)
	try:
		w = stats.wilcoxon(x, y).pvalue
	except ValueError:
		w = 1.0
	return {"a": a, "b": b, "metric": metric, "n": int(len(x)),
		"mean_a": round(float(x.mean()), 4),
		"mean_b": round(float(y.mean()), 4),
		"wins": int((x > y).sum()), "losses": int((x < y).sum()),
		"t_p": float(t.pvalue), "wilcoxon_p": float(w)}


def prf(lab):
	tp = int(((lab["pred"] == 1) & (lab["label"] == 1)).sum())
	fp = int(((lab["pred"] == 1) & (lab["label"] == 0)).sum())
	fn = int(((lab["pred"] == 0) & (lab["label"] == 1)).sum())
	tn = int(((lab["pred"] == 0) & (lab["label"] == 0)).sum())
	p = tp / max(tp + fp, 1)
	r = tp / max(tp + fn, 1)
	return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
		"precision": round(p, 3), "recall": round(r, 3),
		"f1": round(2 * p * r / max(p + r, 1e-9), 3),
		"accuracy": round((tp + tn) / max(len(lab), 1), 3),
		"n": len(lab)}


def term_eval(eng):
	path = os.path.join(DATA, "term_labels.csv")
	if not os.path.exists(path):
		return None
	lab = pd.read_csv(path)
	lab["term"] = lab["term"].str.lower()
	lab["pred"] = lab["term"].isin(set(eng.gloss)).astype(int)
	out = {s: prf(lab[lab["set"] == s]) for s in lab["set"].unique()}
	sample = lab[lab["set"].isin(["random", "reference", "rejected"])]
	out["combined"] = prf(sample.drop_duplicates("term"))
	p = out["random"]["precision"]
	r = out["reference"]["recall"]
	out["estimate"] = {"precision": p, "recall": r,
			   "f1": round(2 * p * r / max(p + r, 1e-9), 3)}
	path = os.path.join(DATA, "origin_labels.csv")
	if os.path.exists(path):
		o = pd.read_csv(path)
		on = o[o["origin_on_arxiv"] == 1]
		out["origin"] = {"named": len(o),
				 "correct": int(o["correct"].sum()),
				 "on_arxiv": len(on),
				 "accuracy_all": round(o["correct"].mean(), 3),
				 "accuracy_on_arxiv":
					round(on["correct"].mean(), 3)}
	return out


def save(name, obj):
	os.makedirs(OUT, exist_ok=True)
	with open(os.path.join(OUT, name), "w") as f:
		json.dump(obj, f, indent=1, default=str)


def fresh_stage(summary):
	eng = Engine("arxiv")
	qs = fresh_queries()
	dev = [q for q in qs if q["split"] == "dev"]
	test = [q for q in qs if q["split"] == "test"]
	print("fresh", len(qs), "queries", len(dev), "dev", len(test), "test")
	cfg, trace = tune(eng, dev)
	eng.cfg.update(cfg)
	save("tuning.json", {"best": cfg, "trace": trace})
	df, pr, lat = run_set(eng, qs, "fresh")
	df.to_csv(os.path.join(OUT, "fresh_runs.csv"), index=False)
	t = df[df["split"] == "test"]
	cols = ["R@5", "R@20", "R@100", "nDCG@10", "MRR", "AP"]
	t = t.assign(has_term=t["terms"] != "")
	sig = []
	for a, b in [("frontier", "hybrid"), ("frontier", "bm25"),
		     ("frontier", "bge"), ("hybrid+E", "hybrid"),
		     ("bm25+E", "bm25"), ("bge+E", "bge"), ("hybrid", "bm25")]:
		for m in ("nDCG@10", "R@20"):
			sig.append(significance(t, a, b, m))
	sub = t[t["has_term"]]
	sig_term = [significance(sub, a, b, "nDCG@10")
		    for a, b in [("frontier", "hybrid"), ("hybrid+E", "hybrid"),
				 ("bm25+E", "bm25"), ("bge+E", "bge"),
				 ("e5+E", "e5")]]
	summary["fresh"] = {
		"cfg": cfg,
		"n": {"all": len(qs), "dev": len(dev), "test": len(test)},
		"test": table(t, cols).reset_index().to_dict("records"),
		"dev": table(df[df["split"] == "dev"], cols).reset_index()
			.to_dict("records"),
		"era": table(t, cols, "era").reset_index().to_dict("records"),
		"has_term": table(t, cols, "has_term").reset_index()
			.to_dict("records"),
		"era_n": t[t["system"] == "bm25"]["era"].value_counts()
			.to_dict(),
		"term_n": int((t[t["system"] == "bm25"]["has_term"]).sum()),
		"sig": sig,
		"sig_term": sig_term,
		"pr": pr,
		"latency_ms": lat,
		"terms": term_eval(eng),
		"glossary": len(eng.gloss),
		"papers": int(pd.read_parquet(os.path.join(
			DATA, "fresh_queries.parquet"))["citing"].nunique()),
		"crawled": len([n for n in
				 os.listdir(os.path.join(DATA, "html"))
				if n.endswith(".gz")]),
	}
	return eng


def lit_stage(summary, cfg):
	eng = Engine("lit")
	eng.cfg.update(cfg)
	qs = lit_queries()
	df, pr, lat = run_set(eng, qs, "litsearch")
	df.to_csv(os.path.join(OUT, "lit_runs.csv"), index=False)
	sig = []
	for a, b in [("hybrid", "bm25"), ("frontier", "hybrid"),
		     ("frontier+pr", "frontier"), ("bge", "bm25")]:
		sig.append(significance(df, a, b, "nDCG@10"))
	tab = lit_table(df)
	summary["lit"] = {
		"table": tab.reset_index().rename(columns={"index": "system"})
			.to_dict("records"),
		"sig": sig, "pr": pr, "latency_ms": lat,
		"n": len(qs),
		"term_queries": int((df[df["system"] == "bm25"]["terms"]
				     != "").sum()),
	}


def main():
	what = sys.argv[1:] or ["fresh", "lit"]
	path = os.path.join(OUT, "summary.json")
	summary = {}
	if os.path.exists(path):
		summary = json.load(open(path))
	for name in ("lit", "arxiv"):
		p = os.path.join(DATA, name, "stats.json")
		if os.path.exists(p):
			ix = summary.setdefault("index", {})
			ix[name] = json.load(open(p))
	if "fresh" in what:
		fresh_stage(summary)
		save("summary.json", summary)
	if "lit" in what:
		cfg = summary.get("fresh", {}).get("cfg", {})
		lit_stage(summary, cfg)
		save("summary.json", summary)
	if "terms" in what:
		from .engine import Engine as E
		summary.setdefault("fresh", {})["terms"] = term_eval(E("arxiv"))
		save("summary.json", summary)


if __name__ == "__main__":
	main()

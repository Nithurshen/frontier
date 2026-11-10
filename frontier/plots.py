import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = os.path.join("results", "figs")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
MUTED = "#b9b8b2"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
NAMES = {
	"tfidf": "TF-IDF", "ql": "QL-Dirichlet", "bm25": "BM25",
	"bm25+prf": "BM25+PRF", "bm25+E": "BM25+E", "e5": "e5-small",
	"e5+E": "e5-small+E", "bge": "bge-small", "bge+E": "bge-small+E",
	"hybrid": "Hybrid", "hybrid+E": "Hybrid+E", "frontier": "Frontier",
	"frontier+pr": "Frontier+PageRank",
}


def style():
	plt.rcParams.update({
		"font.size": 10,
		"axes.edgecolor": GRID,
		"axes.labelcolor": INK2,
		"axes.titlecolor": INK,
		"axes.titlesize": 11,
		"axes.titleweight": "bold",
		"axes.spines.top": False,
		"axes.spines.right": False,
		"axes.grid": True,
		"grid.color": GRID,
		"grid.linewidth": 0.6,
		"xtick.color": INK2,
		"ytick.color": INK2,
		"legend.frameon": False,
		"figure.dpi": 150,
		"savefig.bbox": "tight",
	})


def save(fig, name):
	os.makedirs(OUT, exist_ok=True)
	fig.savefig(os.path.join(OUT, name + ".png"))
	plt.close(fig)


def bars(names, values, focus, title, xlabel, name):
	fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(names) + 1))
	y = np.arange(len(names))
	col = [SERIES[0] if n in focus else MUTED for n in names]
	ax.barh(y, values, color=col, height=0.62, edgecolor="white",
		linewidth=2)
	for i, v in enumerate(values):
		ax.text(v + max(values) * 0.01, i, f"{v:.1f}", va="center",
			fontsize=8, color=INK2)
	ax.set_yticks(y, [NAMES.get(n, n) for n in names])
	ax.invert_yaxis()
	ax.set_xlabel(xlabel)
	ax.set_title(title, loc="left")
	ax.grid(axis="y", visible=False)
	ax.set_xlim(0, max(values) * 1.12)
	save(fig, name)


def grouped(df, groups, systems, title, ylabel, name, labels=None):
	fig, ax = plt.subplots(figsize=(6.4, 3.4))
	w = 0.8 / len(systems)
	x = np.arange(len(groups))
	for k, s in enumerate(systems):
		v = [df.loc[s, g] for g in groups]
		ax.bar(x + (k - (len(systems) - 1) / 2) * w, v, w * 0.92,
		       color=SERIES[k], label=NAMES.get(s, s),
		       edgecolor="white", linewidth=1)
	ax.set_xticks(x, labels or groups)
	ax.set_ylabel(ylabel)
	ax.set_title(title, loc="left")
	ax.grid(axis="x", visible=False)
	ax.legend(ncol=len(systems), loc="upper center",
		  bbox_to_anchor=(0.5, -0.12), fontsize=8)
	save(fig, name)


def pr_curve(pr, systems, title, name):
	fig, ax = plt.subplots(figsize=(5.2, 3.8))
	lv = np.linspace(0, 1, 11)
	for k, s in enumerate(systems):
		ax.plot(lv, pr[s], color=SERIES[k], lw=2, marker="o", ms=4,
			label=NAMES.get(s, s))
	ax.set_xlabel("recall")
	ax.set_ylabel("interpolated precision")
	ax.set_title(title, loc="left")
	ax.legend(fontsize=8)
	save(fig, name)


def zipf(name="zipf"):
	from .index import Index
	ix = Index.load(os.path.join("data", "arxiv", "index.pkl"))
	cf = np.sort(ix.cf)[::-1]
	cf = cf[cf > 0]
	r = np.arange(1, len(cf) + 1)
	fig, ax = plt.subplots(figsize=(5.2, 3.8))
	ax.loglog(r, cf, color=SERIES[0], lw=2, label="arXiv collection")
	ax.loglog(r, cf[0] / r, color=MUTED, lw=1.5, ls="--",
		  label="Zipf: cf ∝ 1/rank")
	ax.set_xlabel("rank")
	ax.set_ylabel("collection frequency")
	ax.set_title("Zipf's law on the arXiv index", loc="left")
	ax.legend(fontsize=8)
	save(fig, name)


def bursts(words, name="bursts"):
	from . import terms
	t = terms.load()
	months = pd.to_datetime(t["months"])
	tot = np.array(t["totals"])
	fig, ax = plt.subplots(figsize=(6.4, 3.6))
	k = 0
	for w in words:
		g = t["terms"].get(w)
		if g is None:
			continue
		rate = np.array(g["counts"]) / tot * 1000
		ax.plot(months, rate, color=SERIES[k], lw=2,
			label=g["surface"])
		start = pd.to_datetime(g["start"])
		ax.axvline(start, color=SERIES[k], lw=1, ls=":")
		k += 1
	ax.set_ylabel("papers per 1,000")
	ax.set_title("Emerging terms and their detected burst start",
		     loc="left")
	ax.legend(fontsize=8)
	save(fig, name)


def frame(rows):
	return pd.DataFrame(rows).set_index("system")


def main():
	style()
	s = json.load(open(os.path.join("results", "summary.json")))
	f = s.get("fresh")
	if f:
		t = frame(f["test"]) * 1
		order = t["nDCG@10"].sort_values(ascending=False).index
		bars(list(order), list(t.loc[order, "nDCG@10"]), {"frontier"},
		     "Fresh-2026 queries, test split", "nDCG@10 (%)",
		     "fresh_ndcg")
		bars(list(order), list(t.loc[order, "R@20"]), {"frontier"},
		     "Fresh-2026 queries, test split", "Recall@20 (%)",
		     "fresh_recall")
		sysm = ["bm25", "bge", "hybrid", "frontier"]
		era = pd.DataFrame(f["era"]).pivot(index="system",
						   columns="era",
						   values="nDCG@10")
		groups = [g for g in ("old", "mid", "new") if g in era.columns]
		labels = {"old": "cited paper\nbefore 2024-07",
			  "mid": "mixed", "new": "cited paper\nafter 2025-07"}
		grouped(era, groups, sysm, "Effect of the cited paper's age",
			"nDCG@10 (%)", "era", [labels[g] for g in groups])
		ht = pd.DataFrame(f["has_term"]).pivot(index="system",
						       columns="has_term",
						       values="nDCG@10")
		grouped(ht, [False, True], sysm,
			"Queries with and without an emerging term",
			"nDCG@10 (%)", "has_term",
			["no emerging term", "contains emerging term"])
		pr_curve(f["pr"], sysm, "Precision-recall, fresh queries",
			 "pr_fresh")
		lat = f["latency_ms"]
		names = [n for n in ("tfidf", "bm25", "ql", "bm25+prf", "bge",
				     "hybrid", "frontier") if n in lat]
		bars(names, [lat[n] for n in names], {"frontier"},
		     "Mean latency per query (302k papers)", "ms", "latency")
	if "lit" in s:
		lt = pd.DataFrame(s["lit"]["table"]).set_index("system")
		pick = ["bm25", "bge", "hybrid", "frontier", "BM25 (paper)",
			"E5-large-v2 (paper)", "Instructor-XL (paper)",
			"GritLM-7B (paper)"]
		pick = [p for p in pick if p in lt.index]
		v = lt.loc[pick, "avg spec R@5"]
		order = v.sort_values(ascending=False).index
		bars(list(order), list(v[order]), {"frontier", "hybrid"},
		     "LitSearch specific questions vs published results",
		     "Recall@5 (%)", "litsearch")
		pr_curve(s["lit"]["pr"], ["bm25", "bge", "hybrid", "frontier"],
			 "Precision-recall, LitSearch", "pr_lit")
	zipf()
	bursts(["grpo", "rlvr", "mcp", "agentic"])
	print("figures in", OUT)


if __name__ == "__main__":
	main()

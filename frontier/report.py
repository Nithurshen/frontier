import json
import os

import pandas as pd

from .plots import NAMES

OUT = "report"
MAIN = ["tfidf", "ql", "bm25", "bm25+prf", "bm25+E", "e5", "e5+E", "bge",
	"bge+E", "hybrid", "hybrid+E", "frontier"]
CORE = ["bm25", "bge", "e5", "hybrid", "hybrid+E", "frontier"]


def f1(x):
	return "--" if x is None or pd.isna(x) else f"{x:.1f}"


def num(x):
	return f"{x:,}" if isinstance(x, int) else str(x)


def pval(p):
	return f"{p:.3f}" if p >= 0.001 else f"{p:.0e}".replace("e-0", "e-")


def name(s):
	return NAMES.get(s, s).replace("+", "{+}")


def bold_best(df, cols):
	out = df.copy().astype(object)
	for c in cols:
		v = pd.to_numeric(df[c], errors="coerce")
		top = v.max()
		out[c] = [("\\textbf{" + f1(x) + "}") if x == top else f1(x)
			  for x in v]
	return out


def tabular(df, cols, align, head):
	lines = ["\\begin{tabular}{" + align + "}", "\\toprule",
		 head + " \\\\", "\\midrule"]
	for idx, r in df.iterrows():
		lines.append(" & ".join([idx] + [str(r[c]) for c in cols]) +
			     " \\\\")
	lines += ["\\bottomrule", "\\end{tabular}"]
	return "\n".join(lines)


def write(name, text):
	os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
	with open(os.path.join(OUT, "tables", name + ".tex"), "w") as f:
		f.write(text + "\n")


def fresh_table(f):
	t = pd.DataFrame(f["test"]).set_index("system")
	t = t.loc[[s for s in MAIN if s in t.index]]
	cols = ["R@5", "R@20", "R@100", "nDCG@10", "MRR", "AP"]
	b = bold_best(t, cols)
	b.index = [name(s) for s in b.index]
	head = "System & R@5 & R@20 & R@100 & nDCG@10 & MRR & MAP"
	write("fresh", tabular(b, cols, "lrrrrrr", head))


def split_table(f, key, cols, labels, fname):
	t = pd.DataFrame(f[key]).pivot(index="system", columns=key,
				       values="nDCG@10")
	t.columns = [str(c) for c in t.columns]
	t = t.loc[[s for s in CORE if s in t.index], cols]
	b = bold_best(t, cols)
	b.index = [name(s) for s in b.index]
	write(fname, tabular(b, cols, "l" + "r" * len(cols),
			     "System & " + " & ".join(labels)))


def lit_table(lit):
	t = pd.DataFrame(lit["table"]).set_index("system")
	ours = ["tfidf", "ql", "bm25", "bm25+prf", "e5", "bge", "hybrid",
		"frontier", "frontier+pr"]
	ours = [s for s in ours if s in t.index]
	paper = [s for s in t.index if s.endswith("(paper)")]
	cols = list(t.columns[:6])
	rows = []
	for s in ours + paper:
		r = t.loc[s, cols]
		cite = " \\cite{ajith2024}"
		label = name(s) if s in ours else s.replace(" (paper)", cite)
		rows.append(" & ".join([label] + [f1(x) for x in r]) + " \\\\")
		if s == ours[-1]:
			rows.append("\\midrule")
	head = ("System & \\multicolumn{2}{c}{Inline-citation} & "
		"\\multicolumn{2}{c}{Author-written} & "
		"\\multicolumn{2}{c}{Average} \\\\\n"
		" & Spec.\\ R@5 & Broad R@20 & Spec.\\ R@5 & Broad R@20 & "
		"Spec.\\ R@5 & Broad R@20")
	text = "\n".join(["\\begin{tabular}{lrrrrrr}", "\\toprule",
			  head + " \\\\", "\\midrule"] + rows +
			 ["\\bottomrule", "\\end{tabular}"])
	write("lit", text)


def sig_table(sig, fname):
	lines = ["\\begin{tabular}{llrrrrr}", "\\toprule",
		 "Comparison & Metric & Mean A & Mean B & W/L & $p_t$ & "
		 "$p_W$ \\\\", "\\midrule"]
	for r in sig:
		lines.append(" & ".join([
			f"{name(r['a'])} vs {name(r['b'])}",
			r["metric"].replace("@", "@"),
			f"{100 * r['mean_a']:.1f}", f"{100 * r['mean_b']:.1f}",
			f"{r['wins']}/{r['losses']}", pval(r["t_p"]),
			pval(r["wilcoxon_p"])]) + " \\\\")
	lines += ["\\bottomrule", "\\end{tabular}"]
	write(fname, "\n".join(lines))


def term_table(t):
	def v(m, k):
		return f"{m[k]:.2f}"
	rows = [("Top-60 detected", t["top"], ["precision"]),
		("Random detected", t["random"], ["precision"]),
		("Reference list", t["reference"], ["recall"]),
		("Rejected", t["rejected"], ["accuracy"]),
		("Combined", t["combined"],
		 ["precision", "recall", "f1", "accuracy"])]
	cols = ["precision", "recall", "f1", "accuracy"]
	lines = ["\\begin{tabular}{lrrrrr}", "\\toprule",
		 "Set & $n$ & P & R & F1 & Acc. \\\\", "\\midrule"]
	for label, m, keep in rows:
		cells = [v(m, c) if c in keep else "--" for c in cols]
		lines.append(" & ".join([label, str(m["n"])] + cells) +
			     " \\\\")
	lines += ["\\bottomrule", "\\end{tabular}"]
	write("terms", "\n".join(lines))


def index_table(ix):
	keys = [("docs", "Documents"), ("terms", "Dictionary terms"),
		("postings", "Postings"), ("tokens", "Tokens (title x2)"),
		("avgdl", "Average document length"),
		("raw_mb", "Postings, 32-bit ids (MB)"),
		("vbyte_mb", "Postings, VB gaps (MB)"),
		("gamma_mb", "Postings, $\\gamma$ gaps (MB)"),
		("zipf_slope", "Zipf slope")]
	lines = ["\\begin{tabular}{lrr}", "\\toprule",
		 "Statistic & arXiv & LitSearch \\\\", "\\midrule"]
	for k, label in keys:
		a, b = ix["arxiv"][k], ix["lit"][k]
		lines.append(f"{label} & {num(a)} & {num(b)} \\\\")
	lines += ["\\bottomrule", "\\end{tabular}"]
	write("index", "\n".join(lines))


def macro(name, value):
	return f"\\newcommand{{\\{name}}}{{{value}}}"


def numbers(s):
	f = s["fresh"]
	t = pd.DataFrame(f["test"]).set_index("system")
	ht = pd.DataFrame(f["has_term"]).pivot(index="system",
					       columns="has_term",
					       values="nDCG@10")
	ht.columns = [str(c) for c in ht.columns]
	lit = pd.DataFrame(s["lit"]["table"]).set_index("system")
	ix = s["index"]
	te = f["terms"]
	m = {
		"NDocs": f"{ix['arxiv']['docs']:,}",
		"NTerms": f"{ix['arxiv']['terms']:,}",
		"NLitDocs": f"{ix['lit']['docs']:,}",
		"NFresh": f["n"]["all"],
		"NDev": f["n"]["dev"],
		"NTest": f["n"]["test"],
		"NTermQ": f["term_n"],
		"NGloss": f["glossary"],
		"NPapers": f.get("papers", "--"),
		"NCrawled": f.get("crawled", "--"),
		"FrontierNdcg": f1(t.loc["frontier", "nDCG@10"]),
		"HybridNdcg": f1(t.loc["hybrid", "nDCG@10"]),
		"BmNdcg": f1(t.loc["bm25", "nDCG@10"]),
		"BgeNdcg": f1(t.loc["bge", "nDCG@10"]),
		"TfidfNdcg": f1(t.loc["tfidf", "nDCG@10"]),
		"FrontierRtw": f1(t.loc["frontier", "R@20"]),
		"HybridRtw": f1(t.loc["hybrid", "R@20"]),
		"BmRtw": f1(t.loc["bm25", "R@20"]),
		"TermFrontier": f1(ht.loc["frontier", "True"]),
		"TermHybrid": f1(ht.loc["hybrid", "True"]),
		"TermBm": f1(ht.loc["bm25", "True"]),
		"TermBmE": f1(ht.loc["bm25+E", "True"]),
		"TermBge": f1(ht.loc["bge", "True"]),
		"TermBgeE": f1(ht.loc["bge+E", "True"]),
		"TermEfive": f1(ht.loc["e5", "True"]),
		"TermEfiveE": f1(ht.loc["e5+E", "True"]),
		"NoTermFrontier": f1(ht.loc["frontier", "False"]),
		"NoTermHybrid": f1(ht.loc["hybrid", "False"]),
		"LitBm": f1(lit.loc["bm25", "avg spec R@5"]),
		"LitHybrid": f1(lit.loc["hybrid", "avg spec R@5"]),
		"LitFrontier": f1(lit.loc["frontier", "avg spec R@5"]),
		"LitBge": f1(lit.loc["bge", "avg spec R@5"]),
		"LitPr": f1(lit.loc["frontier+pr", "avg spec R@5"]),
		"LitBroadHybrid": f1(lit.loc["hybrid", "avg broad R@20"]),
		"LitBroadBm": f1(lit.loc["bm25", "avg broad R@20"]),
		"LitTermQ": s["lit"]["term_queries"],
		"TopP": f"{te['top']['precision']:.2f}",
		"RandP": f"{te['random']['precision']:.2f}",
		"RefR": f"{te['reference']['recall']:.2f}",
		"RejAcc": f"{te['rejected']['accuracy']:.2f}",
		"EstF": f"{te['estimate']['f1']:.2f}",
		"OriginNamed": te["origin"]["named"],
		"OriginCorrect": te["origin"]["correct"],
		"OriginOn": te["origin"]["on_arxiv"],
		"OriginAccOn": f"{100 * te['origin']['accuracy_on_arxiv']:.0f}",
		"LatBm": f["latency_ms"]["bm25"],
		"LatFrontier": f["latency_ms"]["frontier"],
	}
	era = pd.DataFrame(f["era"]).pivot(index="system", columns="era",
					   values="nDCG@10")
	sig = {(r["a"], r["b"], r["metric"]): r for r in f["sig"]}
	fh = sig[("frontier", "hybrid", "nDCG@10")]
	th = {(r["a"], r["b"]): r for r in f["sig_term"]}
	lsig = {(r["a"], r["b"]): r for r in s["lit"]["sig"]}
	lp = lsig[("frontier+pr", "frontier")]
	m.update({
		"SigP": pval(fh["t_p"]),
		"SigWins": fh["wins"],
		"SigLosses": fh["losses"],
		"TermSigP": pval(th[("frontier", "hybrid")]["t_p"]),
		"TermSigWins": th[("frontier", "hybrid")]["wins"],
		"TermSigLosses": th[("frontier", "hybrid")]["losses"],
		"NoTermBm": f1(ht.loc["bm25", "False"]),
		"TermHybridE": f1(ht.loc["hybrid+E", "True"]),
		"HybridENdcg": f1(t.loc["hybrid+E", "nDCG@10"]),
		"EfiveNdcg": f1(t.loc["e5", "nDCG@10"]),
		"EraNewBm": f1(era.loc["bm25", "new"]),
		"EraNewBge": f1(era.loc["bge", "new"]),
		"EraOldBm": f1(era.loc["bm25", "old"]),
		"EraOldBge": f1(era.loc["bge", "old"]),
		"LitPrNdcg": f1(100 * lp["mean_a"]),
		"LitFrontierNdcg": f1(100 * lp["mean_b"]),
		"LitPrP": pval(lp["t_p"]),
		"LatHybrid": f["latency_ms"]["hybrid"],
	})
	keys = {"alpha": "CfgAlpha", "alpha_e": "CfgAlphaE", "beta": "CfgBeta",
		"link": "CfgLink", "k1": "CfgKone", "b": "CfgB"}
	for k, v in keys.items():
		m[v] = f["cfg"][k]
	write("numbers", "\n".join(macro(k, v) for k, v in m.items()))


def main():
	s = json.load(open(os.path.join("results", "summary.json")))
	f = s["fresh"]
	fresh_table(f)
	split_table(f, "era", ["old", "mid", "new"],
		    ["before 2024-07", "mixed", "after 2025-07"], "era")
	split_table(f, "has_term", ["False", "True"],
		    ["no emerging term", "emerging term"], "hasterm")
	sig_table(f["sig"], "sig")
	lit_table(s["lit"])
	sig_table(s["lit"]["sig"], "litsig")
	term_table(f["terms"])
	index_table(s["index"])
	numbers(s)
	print("tables in", os.path.join(OUT, "tables"))


if __name__ == "__main__":
	main()

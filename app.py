import html
import json
import os
import re
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from frontier import boolean, crawl
from frontier.engine import Engine
from frontier.text import stem

st.set_page_config(page_title="Frontier", page_icon="🔭", layout="wide")

SYSTEMS = {
	"Frontier (hybrid + emerging terms + linking)": "frontier",
	"Hybrid + emerging-term expansion": "hybrid+E",
	"Hybrid (BM25 + dense)": "hybrid",
	"BM25": "bm25",
	"BM25 + pseudo-relevance feedback": "bm25+prf",
	"TF-IDF (lnc.ltc)": "tfidf",
	"Query likelihood (Dirichlet)": "ql",
	"Dense (bge-small)": "bge",
	"Dense (e5-small)": "e5",
}
MONTHS = [f"{y}-{m:02d}" for y in range(2023, 2027)
	  for m in range(1, 13) if f"{y}-{m:02d}" <= "2026-11"]
EXAMPLES = [
	"group relative policy optimization for reasoning without a critic",
	"RLVR with verifiable rewards beyond math",
	"benchmarks for coding agents that fix real GitHub issues",
	"Model Context Protocol security of tool-using agents",
	"KV cache compression for long-context inference",
	"vision-language-action models for robot manipulation",
]


@st.cache_resource(show_spinner="Loading index and embeddings...")
def engine(corpus):
	eng = Engine(corpus)
	if corpus == "arxiv":
		eng.tuned = load_cfg()
		eng.cfg.update(eng.tuned)
		eng.ingest(crawl.load())
	else:
		eng.cfg.update(load_cfg())
	return eng


def load_cfg():
	path = os.path.join("results", "summary.json")
	if os.path.exists(path):
		return json.load(open(path)).get("fresh", {}).get("cfg", {})
	return {}


def summary():
	path = os.path.join("results", "summary.json")
	return json.load(open(path)) if os.path.exists(path) else {}


def highlight(text, query, terms):
	stems = {stem(w) for w in re.findall(r"[a-z0-9]+", query.lower())
		 if len(w) > 2}
	out = []
	for tok in re.split(r"(\W+)", text):
		low = tok.lower()
		if low and stem(low) in stems:
			out.append(f"<mark>{html.escape(tok)}</mark>")
		else:
			out.append(html.escape(tok))
	res = "".join(out)
	for t in terms:
		pat = re.compile(r"(?<![\w-])(" + re.escape(t) + r")(?![\w-])",
				 re.I)
		res = pat.sub(r"<b style='color:#7c3aed'>\1</b>", res)
	return res


def snippet(text, n=60):
	words = text.split()
	return " ".join(words[:n]) + (" ..." if len(words) > n else "")


def term_card(eng, t):
	g = eng.gloss[t]
	url = f"https://arxiv.org/abs/{g['origin_id']}"
	st.markdown(
		f"**{g['surface']}**: {g['total']} papers, current burst "
		f"since {g['start']}. Linked to origin paper "
		f"[{g['origin_title']}]({url}) ({g['origin_date']})")
	st.caption(g["definition"])


def show_results(eng, ids, query, found, k, origin):
	for r, i in enumerate(ids[:k]):
		d = eng.doc(i)
		meta = eng.meta.iloc[i]
		if eng.corpus == "arxiv":
			url = f"https://arxiv.org/abs/{d['id']}"
			head = f"**{r + 1}. [{d['title']}]({url})**"
			sub = (f"{d['date']} · {meta['categories']} · "
			       f"{snippet(meta['authors'], 12)}")
		else:
			url = ("https://www.semanticscholar.org/p/" + d["id"])
			head = f"**{r + 1}. [{d['title'].title()}]({url})**"
			sub = f"LitSearch corpus id {d['id']}"
		if i in origin:
			head += "  :violet-badge[origin of an emerging term]"
		if eng.ix.aux is not None and i >= len(eng.ix.dl):
			head += "  :green-badge[crawled today]"
		st.markdown(head)
		st.caption(sub)
		body = highlight(snippet(d["abstract"], 70), query,
				 [eng.gloss[t]["surface"] for t in found])
		st.markdown(f"<div style='font-size:0.92em'>{body}</div>",
			    unsafe_allow_html=True)


def search_tab():
	left, right = st.columns([3, 1])
	with right:
		corpus = st.radio("Corpus", ["arXiv AI papers 2023-2026",
					     "LitSearch (ACL/ICLR)"])
		corpus = "arxiv" if corpus.startswith("arXiv") else "lit"
		label = st.selectbox("Ranking model", list(SYSTEMS))
		k = st.slider("Results", 5, 50, 10)
		since = None
		if corpus == "arxiv":
			since = st.select_slider(
				"Published from",
				options=MONTHS,
				value="2023-01")
		compare = st.toggle("Compare BM25 / dense / Frontier")
	eng = engine(corpus)
	with left:
		pick = st.selectbox("Try an example", [""] + EXAMPLES)
		query = st.text_input("Search the latest AI research",
				      value=pick,
				      placeholder="Describe a paper or idea...")
	if not query:
		st.info("Type a question about recent AI research.")
		return
	t0 = time.time()
	found = eng.spot(query)
	with left:
		if found:
			st.markdown("##### Emerging terms in your query")
			for t in found:
				term_card(eng, t)
	origin = set(eng.links(found))
	if compare:
		cols = st.columns(3)
		for c, s in zip(cols, ["bm25", "bge", "frontier"]):
			ids, _ = eng.search(query, s, k=200)
			ids = filter_since(eng, ids, since)
			with c:
				st.subheader(s)
				show_results(eng, ids, query, found, k, origin)
		return
	ids, _ = eng.search(query, SYSTEMS[label], k=500)
	ids = filter_since(eng, ids, since)
	with left:
		st.caption(f"{len(ids)} candidates · {SYSTEMS[label]} · "
			   f"{1000 * (time.time() - t0):.0f} ms")
		show_results(eng, ids, query, found, k, origin)


def filter_since(eng, ids, since):
	if not since or eng.corpus != "arxiv":
		return ids
	dates = eng.meta["date"].to_numpy()
	return [i for i in ids if dates[i] >= since]


def terms_tab():
	eng = engine("arxiv")
	g = eng.gloss
	if not g:
		st.warning("Run python -m frontier.terms first.")
		return
	rows = [{"term": v["surface"], "key": k, "burst from": v["start"],
		 "burst weight": v["weight"], "papers": v["total"],
		 "origin paper": v["origin_title"]} for k, v in g.items()]
	df = pd.DataFrame(rows)
	q = st.text_input("Filter terms", "")
	if q:
		df = df[df["term"].str.contains(q, case=False, regex=False)]
	left, right = st.columns([2, 3])
	with left:
		st.dataframe(df.drop(columns=["key"]), height=520,
			     hide_index=True)
	with right:
		pick = st.selectbox("Inspect a term", df["key"].tolist(),
				    format_func=lambda x: g[x]["surface"])
		if not pick:
			return
		v = g[pick]
		rate = np.array(v["counts"]) / np.array(eng.totals) * 1000
		chart = pd.DataFrame({"month": pd.to_datetime(eng.months),
				      "papers per 1000": rate})
		line = alt.Chart(chart).mark_line(point=True).encode(
			x="month:T", y="papers per 1000:Q")
		rule = alt.Chart(pd.DataFrame({
			"month": [pd.to_datetime(v["start"])]})).mark_rule(
			color="#7c3aed", strokeDash=[4, 4]).encode(x="month:T")
		st.altair_chart(line + rule, width="stretch")
		st.caption("Dashed line: start of the burst found by "
			   "Kleinberg's two-state automaton.")
		term_card(eng, pick)
		exp = ", ".join(v["expand"])
		st.markdown(f"**Expansion terms:** {exp}")
		st.markdown("**Most recent papers using it**")
		for i in v["docs"][-5:][::-1]:
			d = eng.doc(i)
			st.markdown(f"- [{d['title']}](https://arxiv.org/abs/"
				    f"{d['id']}) ({d['date']})")


def metric_table(rows, cols, index="system"):
	df = pd.DataFrame(rows)
	keep = [c for c in [index] + cols if c in df.columns]
	return df[keep]


def eval_tab():
	s = summary()
	if "fresh" not in s:
		st.warning("Run python -m frontier.evaluate first.")
		return
	f = s["fresh"]
	st.markdown(f"#### Fresh-2026 citation queries (test split, "
		    f"{f['n']['test']} queries)")
	cols = ["R@5", "R@20", "R@100", "nDCG@10", "MRR", "AP"]
	st.dataframe(metric_table(f["test"], cols), hide_index=True)
	a, b = st.columns(2)
	with a:
		st.markdown("**By age of the cited paper**")
		st.dataframe(pd.DataFrame(f["era"]).pivot(
			index="system", columns="era", values="nDCG@10"))
	with b:
		st.markdown("**Queries with vs without an emerging term**")
		st.dataframe(pd.DataFrame(f["has_term"]).pivot(
			index="system", columns="has_term", values="nDCG@10"))
	st.markdown("**Significance (paired t-test, Wilcoxon)**")
	st.dataframe(pd.DataFrame(f["sig"]), hide_index=True)
	if f.get("terms"):
		st.markdown("**Emerging-term detection vs manual labels**")
		st.json(f["terms"])
	if "lit" in s:
		st.markdown(f"#### LitSearch ({s['lit']['n']} queries) "
			    "against the published results")
		st.dataframe(pd.DataFrame(s["lit"]["table"]), hide_index=True)
	figs = os.path.join("results", "figs")
	if os.path.isdir(figs):
		names = [n for n in os.listdir(figs) if n.endswith(".png")]
		names.sort()
		for i in range(0, len(names), 2):
			cs = st.columns(2)
			for c, n in zip(cs, names[i:i + 2]):
				c.image(os.path.join(figs, n), caption=n[:-4])


def index_tab():
	corpus = st.radio("Index", ["arxiv", "lit"], horizontal=True)
	eng = engine(corpus)
	ix = eng.ix
	c = st.columns(5)
	c[0].metric("Documents", f"{ix.n:,}")
	c[1].metric("Terms", f"{len(ix.terms):,}")
	c[2].metric("Postings", f"{ix.post.nnz:,}")
	c[3].metric("Aux docs", f"{len(ix.aux_dl):,}")
	c[4].metric("Merges", ix.merges)
	path = os.path.join("data", corpus, "stats.json")
	if os.path.exists(path):
		st.json(json.load(open(path)))
	st.markdown("#### Boolean retrieval")
	st.caption('AND, OR, NOT, parentheses and "phrases" '
		   '(phrases are conjunctions of their words).')
	q = st.text_input("Boolean query", '(grpo OR rlvr) AND "reward model"')
	if q:
		try:
			ids, trace = boolean.evaluate(ix, q)
		except boolean.QueryError as e:
			st.error(str(e))
			return
		st.caption(" → ".join(f"{t}: {n}" for t, n in trace))
		st.write(f"{len(ids):,} matching documents")
		for i in ids[-10:][::-1]:
			d = eng.doc(i)
			st.markdown(f"- {d['title']} ({d['date']})")
	st.markdown("#### Postings for a term")
	w = st.text_input("Term", "agentic")
	if w:
		t = ix.tid(stem(w.lower()))
		if t < 0:
			st.write("not in the dictionary")
		else:
			df = int(ix.df[t])
			cf = int(ix.cf[t])
			st.write(f"stem `{ix.terms[t]}`: df = {df:,}, "
				 f"idf = {ix.idf[t]:.3f}, cf = {cf:,}")


def crawl_tab():
	st.markdown("Fetch today's new submissions from arXiv listing pages. "
		    "robots.txt is checked and its 15-second crawl delay "
		    "is respected. New papers go to an auxiliary index "
		    "that is merged into the main index once it grows "
		    "large (dynamic indexing).")
	eng = engine("arxiv")
	cats = st.multiselect("Categories", list(crawl.CATS),
			      default=["cs.IR"])
	if st.button("Crawl now"):
		box = st.empty()
		lines = []

		def log(m):
			lines.append(m)
			box.code("\n".join(lines))
		new = crawl.crawl(cats, log)
		n = eng.ingest(new)
		st.success(f"Indexed {n} new papers. "
			   f"Auxiliary index: {len(eng.ix.aux_dl)} docs, "
			   f"merges so far: {eng.ix.merges}.")
	df = crawl.load()
	if len(df):
		st.dataframe(df[["id", "date", "title", "categories"]],
			     hide_index=True)


def main():
	st.title("🔭 Frontier")
	st.caption("Emerging-term-aware search for the latest AI research")
	tabs = st.tabs(["Search", "Emerging terms", "Evaluation", "Index",
			"Crawler"])
	with tabs[0]:
		search_tab()
	with tabs[1]:
		terms_tab()
	with tabs[2]:
		eval_tab()
	with tabs[3]:
		index_tab()
	with tabs[4]:
		crawl_tab()


main()

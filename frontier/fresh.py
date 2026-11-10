import gzip
import os
import re
import sys

import pandas as pd
from bs4 import BeautifulSoup

from .text import sentences
from .web import Fetcher

DATA = "data"
HTML = os.path.join(DATA, "html")
CATS = ("cs.CL", "cs.IR", "cs.LG", "cs.AI")
FROM = "2026-09-01"
SAMPLE = 420
NEW = "2025-07-01"
OLD = "2024-07-01"
MARK = re.compile(r"\[\[(\d+)\]\]")
JUNK = re.compile(r"[<>{}|\\_^=@#]|\n|https?:|Figure|Table|Appendix|"
		  r"Section|Eq\.|Algorithm")
ARXIV = re.compile(r"(?:arxiv|abs/)[:\s/]*(\d{4}\.\d{4,5})", re.I)


def norm(s):
	return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def citing(df):
	prim = df["categories"].str.split().str[0]
	pick = df[(df["date"] >= FROM) & prim.isin(CATS)]
	return pick.sample(min(SAMPLE, len(pick)), random_state=0)


def crawl(df):
	os.makedirs(HTML, exist_ok=True)
	web = Fetcher()
	todo = citing(df)
	for k, aid in enumerate(todo["id"]):
		path = os.path.join(HTML, aid + ".html.gz")
		if os.path.exists(path) or os.path.exists(path + ".miss"):
			continue
		html, info = web.get(f"https://arxiv.org/html/{aid}")
		if html is None or "ltx_bibitem" not in html:
			open(path + ".miss", "w").write(info or "")
			print(k, aid, "miss", info, flush=True)
			continue
		with gzip.open(path, "wt") as f:
			f.write(html)
		print(k, aid, "ok", len(html), flush=True)


def bib_map(soup, titles, ids):
	out = {}
	for li in soup.select("li.ltx_bibitem"):
		text = li.get_text(" ", strip=True)
		m = ARXIV.search(text)
		aid = None
		if m and m.group(1) in ids:
			aid = m.group(1)
		else:
			blocks = li.select("span.ltx_bibblock")
			if len(blocks) >= 2:
				aid = titles.get(norm(blocks[1].get_text(" ")))
		out[li.get("id")] = aid
	return out


def clean(s):
	s = re.sub(r"\s*\[\[\d+\]\]\s*", " ", s)
	s = re.sub(r"\(\s*[,;]?\s*\)|\[\s*[,;]?\s*\]", "", s)
	s = re.sub(r"\s+([,.;:])", r"\1", s)
	s = re.sub(r"([,;])\s*([,;.])", r"\2", s)
	s = re.sub(r"\s+", " ", s).strip(" ,;")
	return s


def contexts(soup, bib):
	for m in soup.select("math"):
		alt = m.get("alttext", "")
		ok = len(alt) < 12 and "\\" not in alt and "{" not in alt
		m.replace_with(f" {alt} " if ok else " [[m]] ")
	groups = []
	for c in soup.select("cite.ltx_cite"):
		keys = [a.get("href", "")[1:] for a in c.select("a")
			if a.get("href", "").startswith("#bib.")]
		c.replace_with(f" [[{len(groups)}]] ")
		groups.append(keys)
	for p in soup.select("p.ltx_p"):
		if p.find_parent(class_="ltx_bibliography"):
			continue
		text = p.get_text(" ", strip=True)
		for s in sentences(text):
			marks = [int(x) for x in MARK.findall(s)]
			if not marks or "[[m]]" in s or JUNK.search(s):
				continue
			keys = [k for g in marks for k in groups[g]]
			yield s, [bib.get(k) for k in keys]


def queries(df):
	titles = dict(zip(df["title"].map(norm), df["id"]))
	ids = set(df["id"])
	date = dict(zip(df["id"], df["date"]))
	rows = []
	for name in sorted(os.listdir(HTML)):
		if not name.endswith(".html.gz"):
			continue
		aid = name[:-8]
		with gzip.open(os.path.join(HTML, name), "rt") as f:
			soup = BeautifulSoup(f.read(), "lxml")
		bib = bib_map(soup, titles, ids)
		seen = set()
		mine = []
		for s, targets in contexts(soup, bib):
			if not targets or None in targets:
				continue
			targets = sorted(set(targets))
			if len(targets) > 2 or aid in targets:
				continue
			if any(date[t] >= date[aid] for t in targets):
				continue
			q = clean(s)
			n = len(q.split())
			if n < 8 or n > 60 or tuple(targets) in seen:
				continue
			if len(re.findall(r"[A-Za-z]", q)) < 0.6 * len(q):
				continue
			seen.add(tuple(targets))
			mine.append((q, targets))
		for q, targets in mine[:4]:
			dates = [date[t] for t in targets]
			if min(dates) >= NEW:
				era = "new"
			elif max(dates) < OLD:
				era = "old"
			else:
				era = "mid"
			rows.append({"query": q, "targets": targets,
				     "citing": aid, "dates": dates, "era": era})
	out = pd.DataFrame(rows)
	out.insert(0, "qid", [f"f{i:04d}" for i in range(len(out))])
	out.to_parquet(os.path.join(DATA, "fresh_queries.parquet"))
	out.assign(valid="").to_csv(
		os.path.join(DATA, "fresh_queries.csv"), index=False)
	print(len(out), "queries from", out["citing"].nunique(), "papers")
	print(out["era"].value_counts().to_string())
	return out


def main():
	df = pd.read_parquet(os.path.join(DATA, "arxiv.parquet"))
	what = sys.argv[1:] or ["crawl", "queries"]
	if "crawl" in what:
		crawl(df)
	if "queries" in what:
		queries(df)


if __name__ == "__main__":
	main()

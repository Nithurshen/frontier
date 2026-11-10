import hashlib
import os
import re
import time

import pandas as pd
from bs4 import BeautifulSoup

from .web import Fetcher

DATA = "data"
CATS = ("cs.IR", "cs.CL", "cs.AI", "cs.LG")
NEW = os.path.join(DATA, "new_papers.parquet")
COLS = ["id", "title", "abstract", "categories", "authors", "date"]


def text(node):
	if node is None:
		return ""
	for d in node.select("span.descriptor"):
		d.decompose()
	s = re.sub(r"\s+", " ", node.get_text(" ", strip=True))
	return re.sub(r" ,", ",", s)


def parse(html, cat):
	soup = BeautifulSoup(html, "lxml")
	out = []
	for dt, dd in zip(soup.select("dl#articles dt"),
			  soup.select("dl#articles dd")):
		a = dt.select_one("a[href^='/abs/']")
		if a is None:
			continue
		aid = re.sub(r"v\d+$", "", a["href"].split("/abs/")[1])
		subj = text(dd.select_one("div.list-subjects"))
		cats = re.findall(r"\(([a-z\-]+\.[A-Z]{2})\)", subj) or [cat]
		out.append({
			"id": aid,
			"title": text(dd.select_one("div.list-title")),
			"abstract": text(dd.select_one("p.mathjax")),
			"categories": " ".join(cats),
			"authors": text(dd.select_one("div.list-authors")),
			"date": time.strftime("%Y-%m-%d"),
		})
	return out


def fingerprint(title, abstract):
	norm = " ".join(re.findall(r"[a-z0-9]+", (title + abstract).lower()))
	return hashlib.sha1(norm.encode()).hexdigest()


def seen():
	ids, prints = set(), set()
	for path in (os.path.join(DATA, "arxiv.parquet"), NEW):
		if os.path.exists(path):
			df = pd.read_parquet(path, columns=["id", "title",
							    "abstract"])
			ids |= set(df["id"])
			if path == NEW:
				prints |= {fingerprint(t, a) for t, a in
					   zip(df["title"], df["abstract"])}
	return ids, prints


def crawl(cats=CATS, log=print):
	web = Fetcher()
	ids, prints = seen()
	fresh = []
	for cat in cats:
		url = f"https://arxiv.org/list/{cat}/new"
		html, info = web.get(url)
		if html is None:
			log(f"{cat}: skipped ({info})")
			continue
		rows = parse(html, cat)
		added = 0
		for r in rows:
			fp = fingerprint(r["title"], r["abstract"])
			if r["id"] in ids or fp in prints or not r["abstract"]:
				continue
			ids.add(r["id"])
			prints.add(fp)
			fresh.append(r)
			added += 1
		log(f"{cat}: {len(rows)} listed, {added} new")
	if fresh:
		df = pd.DataFrame(fresh, columns=COLS)
		if os.path.exists(NEW):
			df = pd.concat([pd.read_parquet(NEW), df])
		df.to_parquet(NEW)
	return pd.DataFrame(fresh, columns=COLS)


def load():
	if os.path.exists(NEW):
		return pd.read_parquet(NEW)
	return pd.DataFrame(columns=COLS)


if __name__ == "__main__":
	crawl()

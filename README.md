# Frontier: emerging-term-aware search for the latest AI research

CSD358 Information Retrieval course project.

AI research vocabulary changes every few months: *GRPO*, *RLVR*, *MCP*, *vibe coding* and *agent skills* did not exist when today's embedding models were trained. Frontier is a search engine over 302,070 recent arXiv papers in AI, ML, NLP and IR (January 2023 to October 2026). It finds these new terms automatically and uses them to improve retrieval.

1. **Classic IR core**, built from scratch with NumPy/SciPy: an inverted index with title zone weighting, Porter stemming, TF-IDF (lnc.ltc), BM25, Dirichlet query likelihood, Rocchio-style pseudo-relevance feedback, Boolean retrieval, dynamic indexing (auxiliary index + merge), postings compression statistics (variable byte, gamma) and Zipf's law.
2. **Dense and hybrid retrieval** with two 2023 embedding models (bge-small, e5-small), fused with BM25 by min-max score fusion.
3. **Emerging-term detection (main novelty).** Kleinberg's burst detection over monthly document frequencies, a contrast against an older background corpus (LitSearch, 2017–2024 papers), and a termhood filter.
4. **Term linking.** Each emerging term is linked to the paper that introduced it. Homonyms such as MCP (Model Context Protocol vs. Mass-Conserving Perceptron) are resolved by similarity to the term's usage centroid.
5. **Emerging-term-aware ranking.** Queries containing an emerging term are expanded with terms from the introducing paper (lexical) and its title (dense). The fusion weight shifts towards BM25, and the introducing paper gets a boost.
6. **Fresh-2026 benchmark.** 2026 arXiv papers are crawled politely (robots.txt, 15 s crawl delay). Their citation sentences become queries and the cited papers become the relevant documents, following the inline-citation method of LitSearch. Search is time-aware: only papers older than the citing paper count.

## Results

Fresh-2026 test split (216 time-aware citation queries; all parameters tuned on the 210 dev queries):

| System | R@5 | R@20 | nDCG@10 | MRR |
|---|---|---|---|---|
| TF-IDF (lnc.ltc) | 41.4 | 53.2 | 42.1 | 42.6 |
| BM25 (k1=1.5, b=0.9) | 42.4 | 50.9 | 40.9 | 41.2 |
| bge-small (dense) | 39.1 | 48.4 | 36.3 | 36.9 |
| Hybrid (BM25 + bge) | 46.3 | 57.4 | 45.0 | 46.1 |
| Hybrid + emerging-term expansion | 47.7 | 57.9 | 45.8 | 46.4 |
| **Frontier** (+ origin linking) | **49.5** | **60.0** | **47.8** | **48.5** |

- Frontier vs. hybrid: +2.8 nDCG@10 overall (paired t-test p = 0.013). On the 47 queries that contain an emerging term the gain is 27.4 → 40.0 (p = 0.011). On the other queries the two are identical.
- LitSearch (EMNLP 2024), recall@5 on specific questions: our BM25 scores 55.8 (paper: 50.0) and our hybrid 64.1. The paper reports E5-large-v2 at 56.2, Instructor-XL at 52.3 and GritLM-7B at 74.8.
- Term detection: precision 0.61 on the top 60 terms, 0.22 on a random tail sample; recall 0.65 on a reference list. Origin linking is correct for 77% of the named artefacts whose introducing paper is on arXiv.

## Setup

Requirements:

- Python 3.13 (the version used for all reported results).
- About 6 GB of free disk (the Hugging Face download cache is about 3 GB; the generated `data/` folder is about 1 GB) and 16 GB of RAM.
- A GPU is optional but strongly recommended for step 2. Frontier picks Apple MPS or CUDA automatically and falls back to the CPU. Embedding 302k papers with two models took roughly 30 to 60 minutes on an Apple M4 Pro. Expect many hours on a CPU.
- Internet access to Hugging Face (step 1). Access to arxiv.org is needed only for the optional crawler.
- Optional: [Tectonic](https://tectonic-typesetting.github.io) (`brew install tectonic`) to rebuild the PDF report.

```bash
git clone https://github.com/Nithurshen/frontier.git
cd frontier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The data, indexes and embeddings are **not** in the repository. Follow the steps below to create them. Only a few small files are committed: the Fresh-2026 queries, the LitSearch queries and the two manual label files.

## Reproduce

Run everything from the repository root with the virtual environment active.

| Step | Command | What it does | Creates | Time (M4 Pro) |
|---|---|---|---|---|
| 1 | `python -m frontier.fetch` | Downloads LitSearch and the arXiv metadata snapshot from Hugging Face (pinned revisions, about 3 GB), then keeps papers in cs.CL, cs.LG, cs.AI or cs.IR first submitted since 2023-01-01 | `data/arxiv.parquet`, `data/lit_corpus.parquet`, `data/lit_queries.parquet` | 10 to 20 min, download-bound |
| 2 | `python -m frontier.build` | Builds both inverted indexes and the bge-small and e5-small embeddings | `data/{lit,arxiv}/index.pkl`, `stats.json`, `dense_bge.npy`, `dense_e5.npy` | 30 to 60 min |
| 3 | `python -m frontier.graph` | PageRank on the LitSearch citation graph | `data/lit/pagerank.npy` | seconds |
| 4 | `python -m frontier.terms` | Detects emerging terms, links them to origin papers, builds the glossary | `data/arxiv/terms.pkl`, `terms.csv` | about 1 min |
| 5 | (nothing) | The 426 Fresh-2026 queries are already committed as `data/fresh_queries.parquet`. See "Rebuilding Fresh-2026" below if you want to crawl them again | | |
| 6 | `python -m frontier.evaluate` | Tunes on the dev split, evaluates all systems on Fresh-2026 and LitSearch, runs significance tests and term-detection scoring | `results/summary.json`, `tuning.json`, `fresh_runs.csv`, `lit_runs.csv` | about 10 min |
| 7 | `python -m frontier.plots` | Draws all figures | `results/figs/*.png` | seconds |
| 8 | `python -m frontier.report` | Writes the LaTeX tables and number macros from `summary.json` | `report/tables/*.tex` | seconds |
| 9 | `cd report && tectonic main.tex` | Builds the PDF | `report/main.pdf` | seconds |
| 10 | `streamlit run app.py` | The demo | | |

What you need for each goal:

- **Run the demo:** steps 1 to 4.
- **Re-check the reported numbers:** steps 1 to 4, then 6. Steps 7 to 9 refresh the figures and the PDF.

Options:

- `python -m frontier.build lit` or `arxiv` builds one corpus only. Add `index-only` to skip embeddings, or `dense-only` to skip the index. Add `bge` or `e5` to embed with one model only.
- `python -m frontier.evaluate fresh`, `lit` or `terms` runs one stage only. `evaluate` overwrites `results/`, which is committed.

### Checks after each step

| After | You should see |
|---|---|
| 1 | 302,070 arXiv papers dated 2023-01-01 to 2026-10-22; 64,183 LitSearch papers and 597 queries |
| 2 | `data/arxiv/stats.json`: 302,070 docs, 234,984 terms, 26,545,143 postings. `data/lit/stats.json`: 64,183 docs, 85,463 terms |
| 4 | `data/arxiv/terms.csv` with 1,500 terms |
| 6 | The tuned values are alpha 0.5, alpha_e 0.5, beta 0.1, link 0.75, k1 1.5, b 0.9. On the Fresh-2026 test split, Frontier reaches nDCG@10 47.8 and the hybrid 45.0. On LitSearch the hybrid's recall@5 on specific questions is 64.1 |

Dense scores can differ in the last digit across hardware (Apple MPS, CUDA and CPU round half-precision differently), so expect tiny deviations such as 47.8 vs 47.7. Everything else is deterministic.

### Rebuilding Fresh-2026 (optional)

The queries are built from citation sentences in arXiv HTML pages. Because arXiv pages change and some papers have no HTML version, a new crawl gives a different query set than the committed one. **The committed file is the one behind every number in the report.**

```bash
python -m frontier.fresh crawl      # fetch about 420 sampled 2026 papers; resumable
python -m frontier.fresh queries    # parse them into data/fresh_queries.parquet and .csv
```

The crawl respects robots.txt and arXiv's 15 second crawl delay, so it takes hours, and arXiv may throttle it. It saves pages to `data/html/` and skips the ones it already has. Stop and resume it as needed.

### Daily crawler (optional)

`python -m frontier.crawl` (or the Crawler tab in the app) fetches today's new cs.IR, cs.CL, cs.AI and cs.LG submissions from the arXiv listing pages. It stores them in `data/new_papers.parquet`. The app adds them to the dynamic auxiliary index at start-up.

### Troubleshooting

- `ModuleNotFoundError`: activate the virtual environment (`source .venv/bin/activate`) and run commands as `python -m frontier.<module>` from the repository root.
- Slow or rate-limited downloads: set a free Hugging Face token (`export HF_TOKEN=...`) and re-run step 1. Finished files are cached and skipped.
- Out of memory while embedding: lower the `batch` default in `encode_docs` in `frontier/dense.py`.
- `build` or `terms` complains about a missing file: run the earlier steps first. Step 4 needs `dense_bge.npy` from step 2, and step 6 needs steps 1 to 4.

## Data

| Data | Source | Used for |
|---|---|---|
| arXiv metadata | [librarian-bots/arxiv-metadata-snapshot](https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot), revision `c428226b689a0e17db2cd55265133b9eca15d97f` (27 Oct 2026) | The 302,070-paper search corpus |
| LitSearch | [princeton-nlp/LitSearch](https://huggingface.co/datasets/princeton-nlp/LitSearch), revision `9573fb284a1026c998df47024b888a163f0f0e25` | 597 queries and 64,183 papers: comparison with the published results, background corpus for term contrast, citation graph for PageRank |
| Fresh-2026 queries | Built by `frontier/fresh.py` from arXiv HTML pages, committed as `data/fresh_queries.parquet` | Time-aware citation-context benchmark (426 queries from 146 papers) |
| Term labels | `data/term_labels.csv` | Precision, recall and F1 of emerging-term detection |
| Origin labels | `data/origin_labels.csv` | Accuracy of linking a term to its introducing paper |

The Hugging Face revisions are pinned in `frontier/fetch.py`, because the arXiv snapshot is refreshed over time and a newer one would contain more papers. Both datasets are used under their own licenses; see their dataset cards.

Layout of the generated `data/` folder:

```
data/
  arxiv.parquet  lit_corpus.parquet  lit_queries.parquet     step 1
  fresh_queries.parquet  fresh_queries.csv                    committed
  term_labels.csv  origin_labels.csv                          committed
  lit/    index.pkl  stats.json  dense_bge.npy  dense_e5.npy  pagerank.npy
  arxiv/  index.pkl  stats.json  dense_bge.npy  dense_e5.npy  terms.pkl  terms.csv
  html/   (only if you rebuild Fresh-2026)
  new_papers.parquet   (only if you run the daily crawler)
```

## Code map

| File | IR role |
|---|---|
| `frontier/fetch.py` | Download and filter the datasets |
| `frontier/text.py` | Tokenisation, stop words, Porter stemming, n-grams |
| `frontier/index.py` | Inverted index (CSR postings), zone weighting, dynamic auxiliary index + merge, compression and Zipf statistics |
| `frontier/rank.py` | TF-IDF lnc.ltc, BM25, query likelihood (Dirichlet), pseudo-relevance feedback, top-k, score fusion |
| `frontier/dense.py` | Dense encoders (bge-small, e5-small) |
| `frontier/boolean.py` | Boolean query parser (AND/OR/NOT, parentheses, phrases) |
| `frontier/terms.py` | Emerging-term detection (Kleinberg bursts, background contrast, termhood) and origin linking |
| `frontier/engine.py` | The search engine: all systems, the Frontier ranking, time-aware filtering, ingestion |
| `frontier/graph.py` | Citation graph and PageRank prior |
| `frontier/web.py`, `frontier/crawl.py` | Polite crawler (robots.txt, crawl delay), content-seen check, daily listing crawl |
| `frontier/fresh.py` | Builds the Fresh-2026 benchmark from citation contexts |
| `frontier/evaluate.py` | Metrics (R@k, P@k, nDCG@10, MRR, MAP, PR curves), tuning, significance tests, term-detection P/R/F1 |
| `frontier/plots.py`, `frontier/report.py` | Figures and report tables |
| `app.py` | Streamlit demo: search, emerging terms, evaluation, index inspector, crawler |

## License

Apache License 2.0. See [LICENSE](LICENSE).

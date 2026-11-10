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

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.13 on an Apple M4 Pro (48 GB). The embedding models are downloaded from Hugging Face on first use.

## Reproduce

```bash
python -m frontier.fetch              # LitSearch + arXiv metadata snapshot (about 3 GB download)
python -m frontier.build              # inverted indexes + dense embeddings (about 1.5 h on Apple GPU)
python -m frontier.graph              # PageRank prior on the LitSearch citation graph
python -m frontier.terms              # emerging-term detection and linking (about 1 min)
python -m frontier.fresh              # crawl arXiv HTML (slow: 15 s crawl delay) and build fresh queries
python -m frontier.evaluate           # tune on the dev split, evaluate everything
python -m frontier.plots              # figures in results/figs
python -m frontier.report             # fill report/numbers.tex and tables
streamlit run app.py                  # the demo
```

`python -m frontier.crawl` (or the Crawler tab in the app) fetches today's new submissions from arXiv listing pages and adds them to the dynamic index.

## Data

| Data | Source | Use |
|---|---|---|
| arXiv metadata | [librarian-bots/arxiv-metadata-snapshot](https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot), 27 Oct 2026 | 302,070 papers in cs.CL, cs.LG, cs.AI or cs.IR since January 2023 |
| LitSearch | [princeton-nlp/LitSearch](https://huggingface.co/datasets/princeton-nlp/LitSearch) | 597 expert-checked queries, 64,183 papers; comparison with the published results |
| Fresh-2026 queries | built by `frontier/fresh.py` from arXiv HTML | citation-context queries; cited papers are the relevant documents |
| Term labels | `data/term_labels.csv`, `data/origin_labels.csv` | manual labels for term detection (precision/recall/F1) and origin linking |

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

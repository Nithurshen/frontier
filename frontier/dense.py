import os
import time

import numpy as np

MODELS = {
	"bge": ("BAAI/bge-small-en-v1.5",
		"Represent this sentence for searching relevant "
		"passages: ", ""),
	"e5": ("intfloat/e5-small-v2", "query: ", "passage: "),
	"qwen": ("Qwen/Qwen3-Embedding-0.6B",
		 "Instruct: Given a description of a research idea, retrieve "
		 "the scientific paper it refers to\nQuery: ", ""),
}

loaded = {}


def device():
	import torch
	if torch.backends.mps.is_available():
		return "mps"
	if torch.cuda.is_available():
		return "cuda"
	return "cpu"


def model(name):
	if name not in loaded:
		from sentence_transformers import SentenceTransformer
		path = MODELS[name][0]
		loaded[name] = SentenceTransformer(path, device=device())
		loaded[name].max_seq_length = 256
		if device() != "cpu":
			loaded[name] = loaded[name].half()
	return loaded[name]


def doc_text(title, abstract):
	return f"{title}. {abstract}"


def encode_docs(name, titles, abstracts, batch=128):
	m = model(name)
	pre = MODELS[name][2]
	texts = [pre + doc_text(t, a) for t, a in zip(titles, abstracts)]
	order = np.argsort([len(t) for t in texts])
	dim = m.get_embedding_dimension()
	out = np.zeros((len(texts), dim), np.float16)
	step = batch * 20
	t0 = time.time()
	for i in range(0, len(texts), step):
		ids = order[i:i + step]
		v = m.encode([texts[j] for j in ids], batch_size=batch,
			     normalize_embeddings=True, convert_to_numpy=True)
		out[ids] = v.astype(np.float16)
		done = min(i + step, len(texts))
		print(f"  {name} {done}/{len(texts)} "
		      f"{done / (time.time() - t0):.0f} docs/s", flush=True)
	return out


def encode_queries(name, texts):
	m = model(name)
	pre = MODELS[name][1]
	v = m.encode([pre + t for t in texts], batch_size=64,
		     normalize_embeddings=True, convert_to_numpy=True)
	return v.astype(np.float32)


class Dense:
	def __init__(self, name, path):
		self.name = name
		self.path = path
		self.mat = np.load(path).astype(np.float32)
		self.extra = np.zeros((0, self.mat.shape[1]), np.float32)

	def scores(self, qv):
		s = self.mat @ qv
		if len(self.extra):
			s = np.concatenate([s, self.extra @ qv])
		return s

	def add(self, titles, abstracts):
		v = encode_docs(self.name, titles, abstracts)
		self.extra = np.vstack([self.extra, v.astype(np.float32)])


def path_of(corpus, name, root="data"):
	return os.path.join(root, corpus, f"dense_{name}.npy")


def build(corpus, name, titles, abstracts, root="data"):
	path = path_of(corpus, name, root)
	if os.path.exists(path):
		print("skip", path)
		return
	os.makedirs(os.path.dirname(path), exist_ok=True)
	v = encode_docs(name, list(titles), list(abstracts))
	np.save(path, v)

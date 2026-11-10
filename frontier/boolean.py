import re

import numpy as np

from .text import STOP, stem

TOKEN = re.compile(r'\(|\)|"[^"]*"|[^\s()"]+')
PREC = {"OR": 1, "AND": 2, "NOT": 3}


class QueryError(Exception):
	pass


def lex(query):
	out = []
	for t in TOKEN.findall(query):
		up = t.upper()
		if up in ("AND", "OR") or t == ")":
			out.append(up if up in PREC else t)
			continue
		if out and out[-1] not in PREC and out[-1] != "(":
			out.append("AND")
		out.append("NOT" if up == "NOT" else t)
	return out


def postfix(tokens):
	out, ops = [], []
	for t in tokens:
		if t == "(":
			ops.append(t)
		elif t == ")":
			while ops and ops[-1] != "(":
				out.append(ops.pop())
			if not ops:
				raise QueryError("unbalanced )")
			ops.pop()
		elif t in PREC:
			while (ops and ops[-1] != "(" and t != "NOT" and
			       PREC[ops[-1]] >= PREC[t]):
				out.append(ops.pop())
			ops.append(t)
		else:
			out.append(t)
	while ops:
		if ops[-1] == "(":
			raise QueryError("unbalanced (")
		out.append(ops.pop())
	return out


def docs_of(ix, word):
	t = ix.tid(stem(word.lower()))
	if t < 0:
		return np.zeros(0, np.int64)
	parts = [d for d, _ in ix.postings(t)]
	return np.concatenate(parts).astype(np.int64)


def phrase(ix, text):
	words = [w for w in re.findall(r"[a-z0-9]+", text.lower())
		 if w not in STOP]
	if not words:
		return np.zeros(0, np.int64)
	lists = sorted((docs_of(ix, w) for w in words), key=len)
	acc = lists[0]
	for p in lists[1:]:
		acc = np.intersect1d(acc, p, assume_unique=True)
	return acc


def evaluate(ix, query):
	stack = []
	trace = []
	for t in postfix(lex(query)):
		if t == "NOT":
			if not stack:
				raise QueryError("NOT needs an operand")
			a = stack.pop()
			r = np.setdiff1d(np.arange(ix.n), a, assume_unique=True)
		elif t in PREC:
			if len(stack) < 2:
				raise QueryError(f"{t} needs two operands")
			b, a = stack.pop(), stack.pop()
			if t == "AND":
				r = np.intersect1d(a, b, assume_unique=True)
			else:
				r = np.union1d(a, b)
		else:
			text = t.strip('"')
			r = phrase(ix, text) if t.startswith('"') else \
				docs_of(ix, text)
		trace.append((t, len(r)))
		stack.append(r)
	if len(stack) != 1:
		raise QueryError("malformed query")
	return stack[0], trace

import re

from nltk.stem import PorterStemmer

STOP = set("""
a about above after again against all also am an and any are as at be
because been before being below between both but by can could did do does
doing down during each few for from further had has have having he her here
hers herself him himself his how i if in into is it its itself just me more
most my myself no nor not now of off on once only or other our ours
ourselves out over own same she should so some such than that the their
theirs them themselves then there these they this those through to too
under until up very was we were what when where which while who whom why
will with would you your yours yourself yourselves via using use used
based paper propose proposed show shows study approach method methods
results new however thus may might many much one two well within without
across among towards toward e g i e et al
""".split())

WORD = re.compile(r"[a-z0-9]+")
SURF = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9\-\.]*[A-Za-z0-9])?")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

stem_cache = {}
porter = PorterStemmer()


def stem(w):
	s = stem_cache.get(w)
	if s is None:
		s = porter.stem(w) if w.isalpha() else w
		stem_cache[w] = s
	return s


def tokens(text, do_stem=True):
	out = []
	for w in WORD.findall(text.lower()):
		if w in STOP or len(w) > 30:
			continue
		out.append(stem(w) if do_stem else w)
	return out


def words(text):
	return SURF.findall(text)


def grams(text, n=3):
	ws = words(text)
	low = [w.lower() for w in ws]
	out = set()
	for i, w in enumerate(low):
		if w in STOP or w.isdigit() or len(w) < 2:
			continue
		out.add(w)
		for k in range(2, n + 1):
			if i + k > len(low):
				break
			span = low[i:i + k]
			if span[-1] in STOP or span[-1].isdigit():
				continue
			if any(x in STOP for x in span):
				continue
			out.add(" ".join(span))
	return out


def sentences(text):
	return [s.strip() for s in SENT.split(text) if s.strip()]

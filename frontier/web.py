import time
import urllib.robotparser
from urllib.parse import urlparse

import requests

AGENT = "FrontierIRCourseCrawler/1.0"
LIMIT = 120


class Fetcher:
	def __init__(self, delay=15.0):
		self.delay = delay
		self.robots = {}
		self.last = {}
		self.http = requests.Session()
		self.http.headers["User-Agent"] = AGENT

	def rules(self, host):
		if host not in self.robots:
			rp = urllib.robotparser.RobotFileParser()
			rp.set_url(f"https://{host}/robots.txt")
			try:
				rp.read()
			except OSError:
				rp = None
			self.robots[host] = rp
		return self.robots[host]

	def allowed(self, url):
		rp = self.rules(urlparse(url).netloc)
		return rp is None or rp.can_fetch(AGENT, url)

	def wait(self, host):
		rp = self.rules(host)
		gap = self.delay
		if rp is not None and rp.crawl_delay(AGENT):
			gap = max(gap, float(rp.crawl_delay(AGENT)))
		spent = time.time() - self.last.get(host, 0)
		if spent < gap:
			time.sleep(gap - spent)
		self.last[host] = time.time()

	def get(self, url):
		if not self.allowed(url):
			return None, "robots"
		self.wait(urlparse(url).netloc)
		start = time.time()
		try:
			r = self.http.get(url, timeout=(15, 30), stream=True)
			body = []
			for part in r.iter_content(1 << 16):
				body.append(part)
				if time.time() - start > LIMIT:
					return None, "slow"
		except requests.RequestException as e:
			return None, type(e).__name__
		if r.status_code != 200:
			return None, str(r.status_code)
		text = b"".join(body).decode(r.encoding or "utf-8", "replace")
		return text, r.url

"""NCI PDQ evidence: every adult and paediatric treatment summary (Health Professional version) from cancer.gov, split into passages with
their section path (NCI text reuse policy https://www.cancer.gov/policies/copyright-reuse: text may be reused with credit to NCI; graphics
excluded; the passages are indexing records, not an unchanged PDQ product, and not clinical practice guidelines). One request every 1.5 s;
HTML is cached with its SHA-256. Writes DATA/library/pdq/{html/*.html, documents.json, passages.jsonl} and the FTS5 database
DATA/library/pdq/pdq.sqlite (tables passages, passages_fts) read by reticle.agent.consult.
  python3 -m reticle.library.pdq        (needs beautifulsoup4 and internet access)"""
import hashlib
import json
import re
import sqlite3
import time
import urllib.parse as up
import urllib.request

from bs4 import BeautifulSoup

from reticle.paths import DATA

D = (DATA/'library/pdq')
UA = {'User-Agent': 'Mozilla/5.0 (RETICLE research prototype)'}
INDEX = ['https://www.cancer.gov/publications/pdq/information-summaries/adult-treatment', 'https://www.cancer.gov/publications/pdq/information-summaries/pediatric-treatment']


def get(url):
    for t in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception:
            time.sleep(5*(t+1))
    raise RuntimeError('failed '+url)


def passages(html, url):
    soup = BeautifulSoup(html, 'html.parser'); article = soup.find('article') or soup.find('main')
    title = soup.find('h1').get_text(' ', strip=True); dates = [x.get('datetime') for x in soup.find_all('time') if x.get('datetime')]
    heads, out = {}, []
    for tag in article.find_all(['h2', 'h3', 'h4', 'h5', 'p', 'li', 'table']):
        text = tag.get_text(' ', strip=True)
        if tag.name.startswith('h'):
            level = int(tag.name[1]); heads = {k: v for k, v in heads.items() if k < level}; heads[level] = text; continue
        if tag.find_parent(['p', 'li', 'table']) is not None or len(text) < 25 or tag.find_parent(['nav', 'aside', 'footer']) is not None:
            continue
        section = [heads[k] for k in sorted(heads)]
        if any(re.match(r'(References|Latest Updates|About This PDQ Summary|Changes to This Summary)', h) for h in section):
            continue  # citation lists and administrative sections, at any heading level
        if re.search(r'et al\.?:|^[A-Z][A-Za-z\'\-]+ [A-Z]{1,3}(, [A-Z][A-Za-z\'\-]+ [A-Z]{1,3})+[,:]', text):
            continue  # a reference-list entry ("Name A, Name B, et al.: Title. Journal ...")
        out.append(dict(title=title, section=' > '.join(section), text=text, source_url=url, source_dates=dates))
    return title, out


def main():
    (D/'html').mkdir(parents=True, exist_ok=True); urls = []
    for ix in INDEX:
        html = get(ix).decode('utf-8', 'replace')
        for h in re.findall(r'href="([^"]*/hp/[^"]*treatment-pdq)"', html):
            u = up.urljoin('https://www.cancer.gov', h)
            if u not in urls:
                urls.append(u)
        time.sleep(1.5)
    print('summaries', len(urls), flush=True); docs, allp = [], []
    for u in urls:
        name = '-'.join(u.rstrip('/').split('/hp/')[1].split('/')); f = D/'html'/f'{name}.html'  # adult/child versions share a page name
        if not f.exists():
            f.write_bytes(get(u)); time.sleep(1.5)
        b = f.read_bytes(); title, ps = passages(b, u)
        for k, p in enumerate(ps, 1):
            p['passage_id'] = f'pdq:{name}:{k}'
        docs.append(dict(name=name, title=title, url=u, sha256=hashlib.sha256(b).hexdigest(), passages=len(ps), source_dates=ps[0]['source_dates'] if ps else []))
        allp += ps; print(name, len(ps), flush=True)
    (D/'documents.json').write_text(json.dumps(dict(source='NCI PDQ treatment summaries, Health Professional versions', policy='https://www.cancer.gov/policies/copyright-reuse',
                                                    retrieved=time.strftime('%Y-%m-%d %H:%M:%S %Z'), note='adapted indexing passages; not clinical practice guidelines', documents=docs), indent=1))
    (D/'passages.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in allp))
    db = D/'pdq.sqlite'; db.unlink(missing_ok=True); con = sqlite3.connect(db)
    con.execute('CREATE TABLE passages(id INTEGER PRIMARY KEY, passage_id TEXT UNIQUE, kind TEXT, title TEXT, section TEXT, text TEXT, source_url TEXT)')
    con.executemany('INSERT INTO passages(passage_id, kind, title, section, text, source_url) VALUES (?,?,?,?,?,?)',
                    [(p['passage_id'], 'pdq', p['title'], p['section'], p['text'], p['source_url']) for p in allp])
    con.execute("CREATE VIRTUAL TABLE passages_fts USING fts5(title, section, text, content='passages', content_rowid='id', tokenize='porter unicode61')")
    con.execute("INSERT INTO passages_fts(passages_fts) VALUES('rebuild')"); con.commit()
    print('documents', len(docs), 'passages', len(allp), 'db', db, flush=True)


if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()   # no options; --help shows the docstring
    main()

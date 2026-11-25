import json, collections
rows = [json.loads(l) for l in open('/mnt/d/kucc/results_cache.jsonl', encoding='utf-8') if l.strip()]
c = collections.Counter((r['sid'], r.get('sig')) for r in rows)
for (sid, sig), n in sorted(c.items(), key=lambda x: (str(x[0][1]), x[0][0])):
    sub = [r for r in rows if r['sid'] == sid and r.get('sig') == sig]
    tt = collections.Counter(r.get('ttype') for r in sub)
    avg = sum(r['steps'] for r in sub) / len(sub)
    print(sig, sid, n, dict(tt), 'avg_steps %.1f' % avg, 'max', max(r['steps'] for r in sub))

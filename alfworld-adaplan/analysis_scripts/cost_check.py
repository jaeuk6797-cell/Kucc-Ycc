import json, collections
rows = [json.loads(l) for l in open('/mnt/d/kucc/results_cache.jsonl', encoding='utf-8') if l.strip()]
tot = collections.defaultdict(lambda: [0, 0.0])
for r in rows:
    k = (r['sid'], r.get('sig'))
    tot[k][0] += 1
    tot[k][1] += r.get('cost_usd', 0.0)
for sid, sig in [('adaplan_M1', 'f267d7daf1'), ('adaplan_M1_adapt', '6f518a3357'), ('adaplan_M1', 'e0a3d47077'), ('adaplan_M1_adapt', 'd4e24efe3b'), ('adaplan_M1', '51ea9ccb01')]:
    n, c = tot[(sid, sig)]
    print(sid, sig, n, 'eps  $/ep %.5f' % (c / n if n else 0))
print('cache total cost $%.4f' % sum(v[1] for v in tot.values()))
try:
    s = 0.0
    for l in open('/mnt/d/kucc/cost_calls.jsonl', encoding='utf-8'):
        if l.strip():
            d = json.loads(l)
            s += d.get('cost_usd', d.get('cost', 0.0))
    print('cost_calls total $%.4f' % s)
except Exception as e:
    print('cost_calls err', e)

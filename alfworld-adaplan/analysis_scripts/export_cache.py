import json, csv, collections, sys
rows = [json.loads(l) for l in open('/mnt/d/kucc/results_cache.jsonl', encoding='utf-8') if l.strip()]
print('rows', len(rows))
print('keys', sorted(set().union(*[set(r) for r in rows[:50]])))
c = collections.Counter(r['sid'] for r in rows)
print(c)
out = sys.argv[1] if len(sys.argv) > 1 else None
if out:
    cols = ['sid', 'sig', 'task', 'rep', 'ttype', 'success', 'steps', 'expansions', 'reward', 'in_tokens', 'out_tokens', 'cost_usd']
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(k, '') for k in cols])
    print('written', out)

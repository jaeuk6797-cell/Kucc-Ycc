import sys, collections
from types import SimpleNamespace
sys.path.insert(0, '.')
import run
cache, spent = run.load_cache('/mnt/d/kucc/results_cache.jsonl')
a = SimpleNamespace(model='claude-haiku-5-5', effort='low', split='eval_out_of_distribution', max_steps=100,
                    mock=False, chain=5, chain_count=0, planner_model=None, seed=0, types=None)
ARMS = ['adaplan_M1', 'adaplan_M1_adapt', 'adaplan_M1_adapt_nohint', 'adaplan_M1_hintonly']
D = {}
for arm in ARMS:
    sig = run.signature(a, run.STRATS[arm])
    D[arm] = {(t, rep): row for (sid, t, rep, s), row in cache.items() if sid == arm and s == sig}
    print(arm, len(D[arm]))
S = {arm: {k: int(bool(r['success'])) for k, r in D[arm].items()} for arm in ARMS}
def cmp(x, y):
    keys = sorted(set(S[x]) & set(S[y]))
    n = len(keys); sx = sum(S[x][k] for k in keys); sy = sum(S[y][k] for k in keys)
    b = sum(1 for k in keys if S[y][k] and not S[x][k]); c = sum(1 for k in keys if S[x][k] and not S[y][k])
    d = collections.defaultdict(list)
    for k in keys:
        d[k[0]].append(S[y][k] - S[x][k])
    m, lo, hi = run.task_boot(d, iters=20000)
    up = sum(1 for v in d.values() if sum(v) > 0); dn = sum(1 for v in d.values() if sum(v) < 0)
    print('%-30s n %d | %d vs %d | %+.1f%%p [%+.1f,%+.1f] | b/c %d/%d p=%.3f | tasks +%d -%d =%d'
          % (x[7:] + ' -> ' + y[7:], n, sx, sy, 100 * m, 100 * lo, 100 * hi, b, c, run.mcnemar_exact(b, c), up, dn, len(d) - up - dn))
for x, y in (('adaplan_M1', 'adaplan_M1_adapt'), ('adaplan_M1', 'adaplan_M1_adapt_nohint'), ('adaplan_M1', 'adaplan_M1_hintonly'),
             ('adaplan_M1_hintonly', 'adaplan_M1_adapt'), ('adaplan_M1_adapt_nohint', 'adaplan_M1_adapt')):
    cmp(x, y)
for rep in (0, 1):
    print('rep', rep, {arm[7:]: sum(v for (t, r), v in S[arm].items() if r == rep) for arm in ARMS})
# partial score / games solved if available
for arm in ARMS:
    rs = [r.get('reward') for r in D[arm].values() if r.get('reward') is not None]
    if rs:
        print(arm, 'mean reward %.3f n %d' % (sum(rs) / len(rs), len(rs)))

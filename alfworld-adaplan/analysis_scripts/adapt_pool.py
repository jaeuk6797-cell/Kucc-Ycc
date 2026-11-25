import sys, collections
from types import SimpleNamespace
sys.path.insert(0, '.')
import run

cache, spent = run.load_cache('/mnt/d/kucc/results_cache.jsonl')
SETS = [('seen', 'eval_in_distribution', 40), ('unseen_hardest', 'eval_out_of_distribution', 20)]
HARD = ('pick_heat_then_place_in_recep', 'pick_cool_then_place_in_recep', 'pick_clean_then_place_in_recep')
D = {}   # key (set, task, rep) -> (m1, adapt, ttype)
for name, split, ms in SETS:
    a = SimpleNamespace(model='claude-haiku-5-5', effort='low', split=split, max_steps=ms,
                        mock=False, chain=0, chain_count=0, planner_model=None, seed=0, types=None)
    R = {}
    for n in ('adaplan_M1', 'adaplan_M1_adapt'):
        sig = run.signature(a, run.STRATS[n])
        R[n] = {(t, rep): row for (sid, t, rep, s), row in cache.items() if sid == n and s == sig}
    for k in set(R['adaplan_M1']) & set(R['adaplan_M1_adapt']):
        D[(name,) + k] = (int(bool(R['adaplan_M1'][k]['success'])), int(bool(R['adaplan_M1_adapt'][k]['success'])),
                          R['adaplan_M1'][k].get('ttype'))

def report(label, keys):
    keys = list(keys)
    n = len(keys)
    m1 = sum(D[k][0] for k in keys); ad = sum(D[k][1] for k in keys)
    b = sum(1 for k in keys if D[k][1] and not D[k][0]); c = sum(1 for k in keys if D[k][0] and not D[k][1])
    d = collections.defaultdict(list)
    for k in keys:
        d[(k[0], k[1])].append(D[k][1] - D[k][0])
    m, lo, hi = run.task_boot(d, iters=20000)
    up = sum(1 for v in d.values() if sum(v) > 0); dn = sum(1 for v in d.values() if sum(v) < 0)
    print('%-28s eps %4d tasks %3d | M1 %3d/%d = %.1f%% | ADaPT %3d/%d = %.1f%% | diff %+.1f%%p [%+.1f,%+.1f] | b/c %d/%d McNemar p=%.3f | tasks better %d worse %d same %d'
          % (label, n, len(d), m1, n, 100 * m1 / n, ad, n, 100 * ad / n, 100 * m, 100 * lo, 100 * hi, b, c,
             run.mcnemar_exact(b, c), up, dn, len(d) - up - dn))

print('meter $%.4f' % spent)
for name, _, _ in SETS:
    report(name, [k for k in D if k[0] == name])
report('POOLED (seen+unseen)', D.keys())
report('POOLED hard types only', [k for k in D if D[k][2] in HARD])
report('seen hard types only', [k for k in D if k[0] == 'seen' and D[k][2] in HARD])

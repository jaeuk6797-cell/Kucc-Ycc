import sys, collections
from types import SimpleNamespace
sys.path.insert(0, '.')
import run

a = SimpleNamespace(model='claude-haiku-5-5', effort='low', split='eval_in_distribution', max_steps=40,
                    mock=False, chain=0, chain_count=0, planner_model=None, seed=0, types=None)
cache, spent = run.load_cache('/mnt/d/kucc/results_cache.jsonl')
print('program meter spent (cache + archive) $%.4f' % spent)
names = ['adaplan_M1', 'adaplan_M1_hintonly', 'adaplan_M1_adapt_nohint', 'adaplan_M1_adapt']
R = {}
for n in names:
    sig = run.signature(a, run.STRATS[n])
    R[n] = {(t, rep): int(bool(row['success'])) for (sid, t, rep, s), row in cache.items() if sid == n and s == sig}
keys = set.intersection(*[set(v) for v in R.values()])
print('common episodes', len(keys), {n: len(R[n]) for n in names})


def diff(f):
    d = collections.defaultdict(list)
    for (t, rep) in keys:
        g = lambda n: R[n][(t, rep)]
        d[t].append(f(g))
    m, lo, hi = run.task_boot(d)
    return '%+.1f%%p [%+.1f, %+.1f]' % (100 * m, 100 * lo, 100 * hi)


M1, HO, NH, AD = names
rows = [
    ('hintonly - M1 (hint effect, no decomposition)', lambda g: g(HO) - g(M1)),
    ('nohint - M1 (decomposition effect, no hint)', lambda g: g(NH) - g(M1)),
    ('M1_adapt - M1 (both)', lambda g: g(AD) - g(M1)),
    ('M1_adapt - hintonly (adding decomposition on top of hint)', lambda g: g(AD) - g(HO)),
    ('M1_adapt - nohint (adding hint on top of decomposition)', lambda g: g(AD) - g(NH)),
    ('hintonly - nohint (hint alone vs decomposition alone)', lambda g: g(HO) - g(NH)),
    ('interaction (both - hint) - (decomp - none)', lambda g: (g(AD) - g(HO)) - (g(NH) - g(M1))),
]
for label, f in rows:
    print(label, diff(f))

b = sum(1 for k in keys if R[HO][k] and not R[AD][k])
c = sum(1 for k in keys if R[AD][k] and not R[HO][k])
print('hintonly only success', b, 'adapt only success', c, '(vs M1_adapt)')
b = sum(1 for k in keys if R[HO][k] and not R[NH][k])
c = sum(1 for k in keys if R[NH][k] and not R[HO][k])
print('hintonly only success', b, 'nohint only success', c, '(vs nohint)')

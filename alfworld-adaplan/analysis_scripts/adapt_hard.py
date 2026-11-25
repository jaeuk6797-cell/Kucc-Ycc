import sys, collections, random
from types import SimpleNamespace
sys.path.insert(0, '.')
import run

a = SimpleNamespace(model='claude-haiku-5-5', effort='low', split='eval_out_of_distribution', max_steps=20,
                    mock=False, chain=0, chain_count=0, planner_model=None, seed=0, types=None)
cache, spent = run.load_cache('/mnt/d/kucc/results_cache.jsonl')
R = {}
for n in ('adaplan_M1', 'adaplan_M1_adapt'):
    sig = run.signature(a, run.STRATS[n])
    R[n] = {(t, rep): int(bool(row['success'])) for (sid, t, rep, s), row in cache.items() if sid == n and s == sig}
keys = sorted(set(R['adaplan_M1']) & set(R['adaplan_M1_adapt']))
print('pairs', len(keys), 'meter $%.4f' % spent)
b = sum(1 for k in keys if R['adaplan_M1_adapt'][k] and not R['adaplan_M1'][k])
c = sum(1 for k in keys if R['adaplan_M1'][k] and not R['adaplan_M1_adapt'][k])
print('adapt only success', b, 'base only success', c, 'McNemar exact p=%.3f' % run.mcnemar_exact(b, c))
for rep in range(3):
    ks = [k for k in keys if k[1] == rep]
    m1 = sum(R['adaplan_M1'][k] for k in ks); ad = sum(R['adaplan_M1_adapt'][k] for k in ks)
    print('rep', rep, 'M1', m1, '/', len(ks), 'ADaPT', ad, '/', len(ks), 'diff %+.1f%%p' % (100 * (ad - m1) / len(ks)))
# task-level: tasks where ADaPT mean > / < / = base mean
tm = collections.defaultdict(lambda: [0, 0, 0])
for (t, rep) in keys:
    tm[t][0] += R['adaplan_M1'][(t, rep)]; tm[t][1] += R['adaplan_M1_adapt'][(t, rep)]; tm[t][2] += 1
up = sum(1 for v in tm.values() if v[1] > v[0]); dn = sum(1 for v in tm.values() if v[1] < v[0]); eq = len(tm) - up - dn
print('tasks ADaPT better', up, 'worse', dn, 'same', eq, 'sign test p=%.3f' % run.mcnemar_exact(up, dn))
# finer bootstrap (20000 iters)
d = collections.defaultdict(list)
for (t, rep) in keys:
    d[t].append(R['adaplan_M1_adapt'][(t, rep)] - R['adaplan_M1'][(t, rep)])
m, lo, hi = run.task_boot(d, iters=20000)
print('task bootstrap 20000: %+.1f%%p [%+.2f, %+.2f]' % (100 * m, 100 * lo, 100 * hi))
# rep0 sub-check vs earlier +6.2

import sys, collections
from types import SimpleNamespace
sys.path.insert(0, '.')
import run

cache, spent = run.load_cache('/mnt/d/kucc/results_cache.jsonl')
HARD = ('pick_heat_then_place_in_recep', 'pick_cool_then_place_in_recep', 'pick_clean_then_place_in_recep')
ARMS = ('adaplan_M1', 'adaplan_M2', 'adaplan_M1_adapt', 'adaplan_M1_adapt_nohint', 'adaplan_M1_hintonly')
SETS = [('seen', 'eval_in_distribution', 40, lambda tt: True),
        ('unseen_hard', 'eval_out_of_distribution', 20, lambda tt: tt in HARD),
        ('unseen_easy', 'eval_out_of_distribution', 20, lambda tt: tt not in HARD)]
D = {}  # (arm) -> {(set,task,rep): success}
for arm in ARMS:
    D[arm] = {}
for name, split, ms, ok in SETS:
    a = SimpleNamespace(model='claude-haiku-5-5', effort='low', split=split, max_steps=ms, mock=False,
                        chain=0, chain_count=0, planner_model=None, seed=0, types=None)
    for arm in ARMS:
        sig = run.signature(a, run.STRATS[arm])
        for (sid, t, rep, s), row in cache.items():
            if sid == arm and s == sig and ok(row.get('ttype')):
                D[arm][(name, t, rep)] = int(bool(row['success']))
print('meter $%.4f' % spent)

def cmp(label, x, y, sets):
    keys = [k for k in set(D[x]) & set(D[y]) if k[0] in sets]
    if not keys:
        print('%-34s (no pairs)' % label); return
    n = len(keys); sx = sum(D[x][k] for k in keys); sy = sum(D[y][k] for k in keys)
    b = sum(1 for k in keys if D[y][k] and not D[x][k]); c = sum(1 for k in keys if D[x][k] and not D[y][k])
    d = collections.defaultdict(list)
    for k in keys:
        d[(k[0], k[1])].append(D[y][k] - D[x][k])
    m, lo, hi = run.task_boot(d, iters=20000)
    up = sum(1 for v in d.values() if sum(v) > 0); dn = sum(1 for v in d.values() if sum(v) < 0)
    print('%-34s eps %4d tasks %3d | %d vs %d (%.1f%% -> %.1f%%) | %+.1f%%p [%+.1f,%+.1f] | b/c %d/%d p=%.3f | tasks +%d -%d =%d'
          % (label, n, len(d), sx, sy, 100 * sx / n, 100 * sy / n, 100 * m, 100 * lo, 100 * hi, b, c,
             run.mcnemar_exact(b, c), up, dn, len(d) - up - dn))

easy = {'unseen_easy'}
print('== new set (unseen easy) ==')
for x, y in (('adaplan_M1', 'adaplan_M2'), ('adaplan_M1', 'adaplan_M1_adapt'), ('adaplan_M1', 'adaplan_M1_adapt_nohint'),
             ('adaplan_M1_adapt', 'adaplan_M1_adapt_nohint')):
    cmp('%s -> %s' % (x[7:], y[7:]), x, y, easy)
print('== per set / pooled 3 sets ==')
for lab, sets in (('seen', {'seen'}), ('unseen_hard', {'unseen_hard'}), ('unseen_easy', easy),
                  ('POOLED 3', {'seen', 'unseen_hard', 'unseen_easy'})):
    for x, y in (('adaplan_M1', 'adaplan_M1_adapt'), ('adaplan_M1', 'adaplan_M1_adapt_nohint'),
                 ('adaplan_M1_adapt', 'adaplan_M1_adapt_nohint')):
        cmp('[%s] %s -> %s' % (lab, x[7:], y[7:]), x, y, sets)

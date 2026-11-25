"""ADaPT 실험 분석 스크립트 (API 비용 없음, 캐시 읽기 전용).

실행 (alfworld-adaplan 폴더에서):
    ADAPLAN_OUT=/mnt/d/kucc ./alfenv/bin/python adapt_analysis.py [캐시경로]

출력: 집합별·합산 짝 비교(과제 단위 부트스트랩 20000회, McNemar 정확검정) + 연속 5개 합성 비교.
비교 표기 "x -> y" 는 (y 성공률 - x 성공률)이다.
"""
import collections
import sys
from types import SimpleNamespace

sys.path.insert(0, '.')
import run  # noqa: E402

CACHE = sys.argv[1] if len(sys.argv) > 1 else '/mnt/d/kucc/results_cache.jsonl'
cache, spent = run.load_cache(CACHE)

HARD = ('pick_heat_then_place_in_recep', 'pick_cool_then_place_in_recep', 'pick_clean_then_place_in_recep')
M1, M2, AD, NH, HO = ('adaplan_M1', 'adaplan_M2', 'adaplan_M1_adapt',
                      'adaplan_M1_adapt_nohint', 'adaplan_M1_hintonly')
ARMS = (M1, M2, AD, NH, HO)
# (이름, split, max_steps, 유형 필터)
SETS = [('seen', 'eval_in_distribution', 40, lambda tt: True),
        ('unseen_hard', 'eval_out_of_distribution', 20, lambda tt: tt in HARD),
        ('unseen_easy', 'eval_out_of_distribution', 20, lambda tt: tt not in HARD)]


def load(args, ok=lambda tt: True, name=None):
    """arm -> {(set, task, rep): success(0/1)}"""
    out = {a: {} for a in ARMS}
    for arm in ARMS:
        sig = run.signature(args, run.STRATS[arm])
        for (sid, t, rep, s), row in cache.items():
            if sid == arm and s == sig and ok(row.get('ttype')):
                out[arm][(name, t, rep)] = int(bool(row['success']))
    return out


def merge(parts):
    out = {a: {} for a in ARMS}
    for p in parts:
        for a in ARMS:
            out[a].update(p[a])
    return out


def cmp(S, label, x, y, sets=None):
    keys = [k for k in set(S[x]) & set(S[y]) if sets is None or k[0] in sets]
    if not keys:
        print('%-34s (짝 없음)' % label)
        return
    n = len(keys)
    sx = sum(S[x][k] for k in keys)
    sy = sum(S[y][k] for k in keys)
    b = sum(1 for k in keys if S[y][k] and not S[x][k])
    c = sum(1 for k in keys if S[x][k] and not S[y][k])
    d = collections.defaultdict(list)
    for k in keys:
        d[(k[0], k[1])].append(S[y][k] - S[x][k])
    m, lo, hi = run.task_boot(d, iters=20000)
    up = sum(1 for v in d.values() if sum(v) > 0)
    dn = sum(1 for v in d.values() if sum(v) < 0)
    print('%-34s 에피소드 %4d 과제 %3d | %d -> %d (%.1f%% -> %.1f%%) | %+.1f%%p [%+.1f,%+.1f] | 짝불일치 %d/%d p=%.3f | 과제 +%d -%d =%d'
          % (label, n, len(d), sx, sy, 100 * sx / n, 100 * sy / n, 100 * m, 100 * lo, 100 * hi, b, c,
             run.mcnemar_exact(b, c), up, dn, len(d) - up - dn))


def base_args(split, max_steps, **kw):
    d = dict(model='claude-haiku-5-5', effort='low', split=split, max_steps=max_steps, mock=False,
             chain=0, chain_count=0, planner_model=None, seed=0, types=None)
    d.update(kw)
    return SimpleNamespace(**d)


print('프로그램 계량기 누적 $%.4f  (캐시: %s)' % (spent, CACHE))

# ---- 단일 게임 3개 집합 ----
S = merge([load(base_args(split, ms), ok, name) for name, split, ms, ok in SETS])
PAIRS = [(M1, AD), (M1, HO), (M1, NH), (M1, M2), (HO, AD), (NH, AD), (NH, HO)]
for lab, sets in [('seen', {'seen'}), ('unseen_hard (최난이도)', {'unseen_hard'}),
                  ('unseen_easy (비최난이도)', {'unseen_easy'}),
                  ('합산 3집합', {'seen', 'unseen_hard', 'unseen_easy'})]:
    print('\n== %s' % lab)
    for x, y in PAIRS:
        cmp(S, '%s -> %s' % (x[7:], y[7:]), x, y, sets)

# ---- 연속 5개 합성 (26개 x 반복) ----
C = load(base_args('eval_out_of_distribution', 100, chain=5), name='chain5')
print('\n== 연속 5개 합성 (스텝 100, 5개 모두 성공이 성공)')
for a in (M1, AD, NH, HO):
    n = len(C[a])
    if n:
        print('  %-26s %d/%d = %.1f%%' % (a, sum(C[a].values()), n, 100 * sum(C[a].values()) / n))
for x, y in [(M1, AD), (M1, HO), (M1, NH), (HO, AD), (NH, AD)]:
    cmp(C, '%s -> %s' % (x[7:], y[7:]), x, y)

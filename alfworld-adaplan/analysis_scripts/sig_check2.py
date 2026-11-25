import sys
from types import SimpleNamespace
sys.path.insert(0, '.')
import run
for split, ms in (('eval_in_distribution', 40), ('eval_out_of_distribution', 40)):
    a = SimpleNamespace(model='claude-haiku-5-5', effort='low', split=split, max_steps=ms,
                        mock=False, chain=0, chain_count=0, planner_model=None, seed=0, types=None)
    print(split, ms)
    for sid in ('adaplan_M1', 'adaplan_M1_adapt', 'adaplan_M1_hintonly', 'adaplan_M1_adapt_nohint'):
        print('  ', sid, run.signature(a, run.STRATS[sid]))
print('seen cached: M1 e0a3d47077, M1_adapt d4e24efe3b (280 each)')

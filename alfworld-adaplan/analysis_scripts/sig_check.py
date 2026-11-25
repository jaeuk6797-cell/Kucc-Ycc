import sys
from types import SimpleNamespace
sys.path.insert(0, '.')
import run
a = SimpleNamespace(model='claude-haiku-5-5', effort='low', split='eval_out_of_distribution', max_steps=20,
                    mock=False, chain=0, chain_count=0, planner_model=None, seed=0, types=None)
for sid in ('adaplan_M1', 'adaplan_M1_adapt', 'adaplan_M1_hintonly', 'adaplan_M1_adapt_nohint'):
    print(sid, run.signature(a, run.STRATS[sid]))
print('expected M1 f267d7daf1, M1_adapt 6f518a3357')

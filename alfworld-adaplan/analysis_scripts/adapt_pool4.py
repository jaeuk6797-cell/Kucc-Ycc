import re
src = open('/mnt/c/Users/jaeuk/AppData/Local/Temp/adapt_pool3.py', encoding='utf-8').read()
head = src.split("easy = {'unseen_easy'}")[0]
exec(head)
M1, AD, HO, NH = 'adaplan_M1', 'adaplan_M1_adapt', 'adaplan_M1_hintonly', 'adaplan_M1_adapt_nohint'
pairs = [(M1, AD), (M1, HO), (M1, NH), (HO, AD), (NH, AD), (NH, HO)]
for lab, sets in (('unseen_hard', {'unseen_hard'}), ('seen', {'seen'}), ('unseen_easy', {'unseen_easy'}),
                  ('POOLED seen+hard (hintonly/nohint 있는 집합)', {'seen', 'unseen_hard'}),
                  ('POOLED nohint 3집합', {'seen', 'unseen_hard', 'unseen_easy'})):
    print('==', lab)
    for x, y in pairs:
        cmp('%s -> %s' % (x[7:], y[7:]), x, y, sets)

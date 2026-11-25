src = open('/mnt/c/Users/jaeuk/AppData/Local/Temp/adapt_pool3.py', encoding='utf-8').read()
exec(src.split("easy = {'unseen_easy'}")[0])
M1, AD, NH = 'adaplan_M1', 'adaplan_M1_adapt', 'adaplan_M1_adapt_nohint'
for lab, sets in (('unseen_hard', {'unseen_hard'}), ('seen', {'seen'}), ('unseen_easy', {'unseen_easy'}),
                  ('POOLED 3', {'seen', 'unseen_hard', 'unseen_easy'})):
    print('==', lab)
    cmp('M1 -> nohint', M1, NH, sets)
    for rep in range(6):
        ks = [k for k in set(D[M1]) & set(D[NH]) if k[0] in sets and k[2] == rep]
        if ks:
            print('  rep', rep, 'n', len(ks), 'M1', sum(D[M1][k] for k in ks), 'nohint', sum(D[NH][k] for k in ks))

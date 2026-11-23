import time
from costs import build
SETS = [('asteroids_05', 'jovianmoons_04', 'asteroids_10', 'jovianmoons_10', 'asteroids_20', 'jovianmoons_20', 'asteroids_40', 'asteroids_80')]
for nT in (6, 11, 21, 41, 81, 161):
    for name in SETS[0]:
        n = int(name.split('_')[1])
        if nT <= n:   # ACT requires |T| > |A|
            continue
        t = time.time(); build(name, nT)
        print(f'{name} T{nT} {time.time()-t:.0f}s', flush=True)

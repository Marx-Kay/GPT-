"""Digitise CGM (2005) Figure 2 from the PDF vector layer.

Calibration and legend mapping were read off the page text layer; curve points are
reconstructed by chaining PDF drawing paths that share the same line width and whose
endpoints coincide.
"""
import numpy as np, pymupdf, json, collections, itertools

pg = pymupdf.open("/tmp/gpt-repo/cgm.pdf")[11]

XT = [(0, 95.55), (25, 116.6), (50, 137.7), (75, 158.8), (100, 179.85), (125, 200.95),
      (150, 222.05), (175, 243.15), (200, 264.25), (225, 285.35), (250, 306.45),
      (275, 327.5), (300, 348.55)]
YTB = [(1.0, 233.01), (0.8, 254.89), (0.6, 277.06), (0.4, 298.94), (0.2, 321.11), (0.0, 343.36)]
YTC = [(50, 381.65), (40, 407.15), (30, 432.65), (20, 457.85), (10, 483.35), (0, 508.85)]
YTA = [(1.0, 68.00), (0.8, 91.07), (0.6, 114.14), (0.4, 137.22), (0.2, 159.99), (0.0, 183.06)]


def fit(t):
    c = np.array([a for _, a in t], float); v = np.array([b for b, _ in t], float)
    return np.polyfit(c, v, 1)


AX, AY_A, AY_B, AY_C = fit(XT), fit(YTA), fit(YTB), fit(YTC)
fx = lambda p: AX[0] * p + AX[1]
fyA = lambda p: AY_A[0] * p + AY_A[1]
fyB = lambda p: AY_B[0] * p + AY_B[1]
fyC = lambda p: AY_C[0] * p + AY_C[1]

paths = pg.get_drawings()


def seg_points(p):
    pts = []
    for it in p["items"]:
        if it[0] == "l":
            pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
        elif it[0] == "c":
            pts += [(it[1].x, it[1].y), (it[2].x, it[2].y), (it[3].x, it[3].y), (it[4].x, it[4].y)]
    return pts


def inside(r, y0, y1):
    return r.y0 >= y0 and r.y1 <= y1 and r.width > 0.3


def chain(paths_with_pts, tol=1.6):
    """greedy chaining of polylines by endpoint proximity"""
    chains = []
    used = [False] * len(paths_with_pts)
    for i in range(len(paths_with_pts)):
        if used[i]:
            continue
        used[i] = True
        cur = list(paths_with_pts[i])
        changed = True
        while changed:
            changed = False
            for j in range(len(paths_with_pts)):
                if used[j]:
                    continue
                q = paths_with_pts[j]
                d_head = np.hypot(cur[0][0] - q[-1][0], cur[0][1] - q[-1][1])
                d_tail = np.hypot(cur[-1][0] - q[0][0], cur[-1][1] - q[0][1])
                d_hh = np.hypot(cur[0][0] - q[0][0], cur[0][1] - q[0][1])
                d_tt = np.hypot(cur[-1][0] - q[-1][0], cur[-1][1] - q[-1][1])
                m = min(d_head, d_tail, d_hh, d_tt)
                if m < tol:
                    if m == d_tail:
                        cur = cur + q
                    elif m == d_head:
                        cur = q + cur
                    elif m == d_hh:
                        cur = q[::-1] + cur
                    else:
                        cur = cur + q[::-1]
                    used[j] = True
                    changed = True
        chains.append(cur)
    return chains


def build(y0, y1, width_map, age_map):
    """width_map: not used; age_map: {round(width,3): age}"""
    byw = collections.defaultdict(list)
    for p in paths:
        r = p["rect"]
        if not inside(r, y0, y1):
            continue
        w = p.get("width")
        if w is None:
            continue
        byw[round(w, 3)].append(seg_points(p))
    out = {}
    for w, age in age_map.items():
        key = min(byw, key=lambda k: abs(k - w)) if byw else None
        if key is None:
            continue
        ch = chain(byw[key])
        ch.sort(key=lambda c: -len(c))
        out[age] = np.array(ch[0]) if ch else np.zeros((0, 2))
    return out, byw


C_PT, bywC = build(378.0, 512.0, None, {1.114: 65, 0.743: 35, 0.371: 20})
B_PT, bywB = build(228.0, 346.0, None, {1.113: 75, 0.742: 20, 0.371: 30})

XS = [0, 2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 300]
print("=" * 128)
print("CGM (2005) Fig. 2C - consumption policy C(x,t), thousands of 1992 USD   [PDF vector digitisation]")
print("cash-on-hand  :" + "".join("%8d" % v for v in XS))
FIGC = {}
for age in sorted(C_PT):
    P = C_PT[age]; x = fx(P[:, 0]); y = fyC(P[:, 1])
    o = np.argsort(x); x, y = x[o], y[o]
    FIGC[age] = np.c_[x, y]
    row = [np.interp(v, x, y) if x[0] <= v <= x[-1] else np.nan for v in XS]
    print("  age %3d     :" % age + "".join("%8.1f" % v for v in row))

print()
print("=" * 128)
print("CGM (2005) Fig. 2B - optimal stock share alpha(x,t)                     [PDF vector digitisation]")
print("cash-on-hand  :" + "".join("%8d" % v for v in XS))
FIGB = {}
for age in sorted(B_PT):
    P = B_PT[age]; x = fx(P[:, 0]); y = fyB(P[:, 1])
    o = np.argsort(x); x, y = x[o], y[o]
    FIGB[age] = np.c_[x, y]
    row = [np.interp(v, x, y) if x[0] <= v <= x[-1] else np.nan for v in XS]
    print("  age %3d     :" % age + "".join("%8.2f" % v for v in row))

print()
print("available line widths, panel C:", sorted(bywC.keys()))
print("available line widths, panel B:", sorted(bywB.keys()))
json.dump(dict(C={str(k): v.tolist() for k, v in FIGC.items()},
               B={str(k): v.tolist() for k, v in FIGB.items()}),
          open("/tmp/fig2_targets.json", "w"), indent=1)
print("saved /tmp/fig2_targets.json")

import numpy as np, pymupdf, json, collections
pg=pymupdf.open("/tmp/gpt-repo/cgm.pdf")[32]
# calibration from the text layer
XT=[(20,84.45),(25,101.95),(30,119.45),(35,137.0),(40,154.05),(45,171.55),(50,189.05),
    (55,206.6),(60,224.1),(65,241.6),(70,258.7),(75,276.2),(80,293.7),(85,311.2),(90,328.7),(95,346.2)]
YT=[(0,187.9),(0.2,164.0),(0.4,139.8),(0.6,115.9),(0.8,91.9),(1.0,68.0)]
fit=lambda t: np.polyfit([c for _,c in t],[v for v,_ in t],1)
AX,AY=fit(XT),fit(YT)
fx=lambda p: AX[0]*p+AX[1]; fy=lambda p: AY[0]*p+AY[1]
paths=pg.get_drawings()
def pts(p):
    o=[]
    for it in p["items"]:
        if it[0]=="l": o+=[(it[1].x,it[1].y),(it[2].x,it[2].y)]
        elif it[0]=="c": o+=[(it[1].x,it[1].y),(it[2].x,it[2].y),(it[3].x,it[3].y),(it[4].x,it[4].y)]
    return o
# legend line samples -> width per rule
leg={}
for p in paths:
    r=p["rect"]
    if 208<r.y0<216 and r.width>3 and p.get("width"):
        lab=min([(114.0,"Optimal"),(157.8,"100-age"),(201.9,"No Income"),(252.8,"No IncomeRisk"),(316.1,"Approx.")],
                key=lambda t: abs(t[0]-(r.x1+1)))
        leg[lab[1]]=round(p["width"],3)
print("legend widths:",leg)
series=collections.defaultdict(list)
for p in paths:
    r=p["rect"]
    if not(66<=r.y0 and r.y1<=190) or r.width<0.5 or p.get("width") is None: continue
    series[round(p["width"],3)]+=pts(p)
print("widths present:",{k:len(v) for k,v in sorted(series.items())})
out={}
for name,w in leg.items():
    k=min(series,key=lambda kk: abs(kk-w))
    P=np.array(series[k]); x=fx(P[:,0]); y=fy(P[:,1])
    o=np.argsort(x); x,y=x[o],y[o]
    out[name]=np.c_[x,y]
    ag=[20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]
    print(f"  {name:12s}:"+"".join("%7.2f"%np.interp(a,x,y) for a in ag))
print("  ages        :"+"".join("%7d"%a for a in [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]))
json.dump({k:v.tolist() for k,v in out.items()},open("/tmp/fig11_targets.json","w"))

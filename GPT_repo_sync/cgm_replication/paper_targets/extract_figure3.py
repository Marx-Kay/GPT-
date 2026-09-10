"""Digitize published PDF vector paths; these are approximate graphical targets."""
from pathlib import Path
import csv, json, hashlib
import fitz
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
source=ROOT/"cgm.pdf"
p=fitz.open(source)[14]
ds=p.get_drawings()
# Anchor every displayed tick to its labelled data value. The plot's top border
# in panel C is 1.1, whereas the labelled 1.0 tick lies at y=397.833.
def anchors(idx, n_y):
    items=ds[idx]["items"]
    y=np.array([item[1].y for item in items[1:n_y+1]])
    x=np.array([item[1].x for item in items[n_y+2:]])
    return x,y
x_a,y_a=anchors(10,6)
x_c,y_c=anchors(535,6)
assert len(x_a)==17 and len(x_c)==16
assert np.all(np.diff(x_a)>0) and np.all(np.diff(y_a)<0)
xa_data=np.arange(20,101,5); ya_data=np.arange(0,251,50)
xc_data=np.arange(20,96,5); yc_data=np.arange(0,1.01,.2)
# Straight coordinate transform fitted to all ticks, rather than image bounds.
# Printed vector positions have <0.3pt typesetting quantization.
fit_a_x=np.polyfit(x_a,xa_data,1); fit_a_y=np.polyfit(y_a,ya_data,1)
fit_c_x=np.polyfit(x_c,xc_data,1); fit_c_y=np.polyfit(y_c,yc_data,1)

def solid(ids):
    pts=[]
    for i in ids:
        for item in ds[i]["items"]:
            assert item[0]=="l"
            for q in item[1:]: pts.append((q.x,q.y,i))
    return sorted(set(pts))

def fill_centers(ids):
    pts=[]
    for i in ids:
        d=ds[i]
        assert d["type"]=="f"
        vertices=[]
        for it in d["items"]:
            if it[0]=="l": vertices += [(it[1].x,it[1].y),(it[2].x,it[2].y)]
            elif it[0]=="re":
                r=it[1];vertices += [(r.x0,r.y0),(r.x0,r.y1),(r.x1,r.y0),(r.x1,r.y1)]
            else: raise AssertionError(it)
        q=np.array(sorted(set(vertices)))
        pts.append((float(q[:,0].mean()),float(q[:,1].mean()),i))
    return pts

specs=[
 ("consumption", "A", fill_centers(range(11,171))),
 ("income", "A", solid([171])),
 ("wealth", "A", fill_centers(range(172,314))),
 ("alpha_mean", "C", solid([536,537])),
 ("alpha_p05", "C", fill_centers(range(538,649))),
 ("alpha_p95", "C", fill_centers(range(649,823))),
]
series={}
raw=[]
for name,panel,points in specs:
    fx,fy=(fit_a_x,fit_a_y) if panel=="A" else (fit_c_x,fit_c_y)
    points=sorted(points)
    ages=[];vals=[]
    for x,y,idx in points:
        age=float(np.polyval(fx,x));v=float(np.polyval(fy,y))
        # The source has small drawing quantization, so retain raw values even
        # when a line is within 0.002 of alpha=1 instead of silently clipping.
        raw.append([name,panel,idx,x,y,age,v]);ages.append(age);vals.append(v)
    series[name]=(np.array(ages),np.array(vals))
with (OUT/"figure3_raw_vector_points.csv").open("w",encoding="utf-8",newline="") as f:
    w=csv.writer(f);w.writerow(["series","panel","drawing_index","pdf_x_pt","pdf_y_pt","age","value"]);w.writerows(raw)
rows=[]
for age in range(20,101):
    row={"age":age}
    for name,(ages,vals) in series.items():
        # At a boundary allow at most .06 years of plot-origin rounding for
        # solid paths. Filled dash centers do not provide endpoint values.
        tol=.06 if name in {"income","alpha_mean"} else 0
        row[name]=float(np.interp(age,ages,vals)) if ages[0]-tol<=age<=ages[-1]+tol else None
    rows.append(row)
with (OUT/"figure3_annual_targets.csv").open("w",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
meta={
 "source_pdf":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
 "pdf_page_1_based":15,"printed_page":505,"figure":"3",
 "money_unit":"thousands of 1992 US dollars",
 "not_original_simulation_data":True,
 "method":"solid paths: vertices; dashed/dotted paths: filled polygon centers; affine axes fitted to all labelled vector ticks; annual linear interpolation inside support",
 "approximate_reading_tolerances":{"wealth_thousand_usd":2.0,"income_consumption_thousand_usd":1.0,"alpha":0.01,"age_years":0.15},
 "panel_A":{"x_tick_pdf_pt":x_a.tolist(),"x_tick_values":xa_data.tolist(),"y_tick_pdf_pt":y_a.tolist(),"y_tick_values":ya_data.tolist(),"x_affine_coefficients":fit_a_x.tolist(),"y_affine_coefficients":fit_a_y.tolist()},
 "panel_C":{"x_tick_pdf_pt":x_c.tolist(),"x_tick_values":xc_data.tolist(),"y_tick_pdf_pt":y_c.tolist(),"y_tick_values":yc_data.tolist(),"x_affine_coefficients":fit_c_x.tolist(),"y_affine_coefficients":fit_c_y.tolist(),"top_border_value":float(np.polyval(fit_c_y,ds[534]["rect"].y0))},
 "series":{name:{"first_age":float(a[0]),"last_age":float(a[-1]),"raw_point_count":len(a),"max_value":float(v.max()),"age_at_max":float(a[v.argmax()]),"min_value":float(v.min()),"age_at_min":float(a[v.argmin()])} for name,(a,v) in series.items()},
}
(OUT/"figure3_extraction_metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
fig,ax=plt.subplots(2,1,figsize=(8,8),layout="constrained")
for name in ["consumption","income","wealth"]:
    a,v=series[name];ax[0].plot(a,v,label=name)
for name in ["alpha_mean","alpha_p05","alpha_p95"]:
    a,v=series[name];ax[1].plot(a,v,label=name)
ax[0].set(ylabel="Thousands of 1992 USD",ylim=(0,250),xlim=(20,100),title="CGM published Figure 3A: PDF vector extraction")
ax[1].set(ylabel="Stock share",xlabel="Age",ylim=(0,1.1),xlim=(20,99),title="CGM published Figure 3C: PDF vector extraction")
for a in ax:a.grid(alpha=.3);a.legend()
fig.savefig(OUT/"figure3_extracted.png",dpi=160)
print(json.dumps({"targets":[r for r in rows if r["age"] in [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99]],"series":meta["series"],"top_C":meta["panel_C"]["top_border_value"]},indent=2))

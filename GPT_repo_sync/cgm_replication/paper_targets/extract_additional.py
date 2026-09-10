"""Extract Fig 2A and Fig 3B graphical targets, preserving source anomalies."""
from pathlib import Path
import csv,json
import fitz
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
doc=fitz.open(ROOT/"cgm.pdf")
def centers(ds,start,stop):
 out=[]
 for i in range(start,stop):
  d=ds[i];assert d["type"]=="f"
  v=[]
  for it in d["items"]:
   if it[0]=="l":v.extend([(q.x,q.y) for q in it[1:]])
   elif it[0]=="re":
    r=it[1];v.extend([(r.x0,r.y0),(r.x0,r.y1),(r.x1,r.y0),(r.x1,r.y1)])
  q=np.array(sorted(set(v)));out.append([*q.mean(axis=0),i])
 return out

def save(name,points,xf,yf,grid,first_col):
 raw=sorted([(np.polyval(xf,x),np.polyval(yf,y),x,y,i) for x,y,i in points])
 with (OUT/(name+"_raw.csv")).open("w",encoding="utf-8",newline="") as f:
  w=csv.writer(f);w.writerow([first_col,"value","pdf_x_pt","pdf_y_pt","drawing_index"]);w.writerows(raw)
 xx,yy=np.array(raw)[:,:2].T
 rows=[(x,float(np.interp(x,xx,yy)) if xx[0]<=x<=xx[-1] else None) for x in grid]
 with (OUT/(name+"_targets.csv")).open("w",encoding="utf-8",newline="") as f:
  w=csv.writer(f);w.writerow([first_col,"value"]);w.writerows(rows)
 return rows
p=doc[11];ds=p.get_drawings();it=ds[10]["items"]
x=np.array([q[1].x for q in it[8:]]);y=np.array([q[1].y for q in it[1:7]])
xf=np.polyfit(x,np.arange(0,301,25),1);yf=np.polyfit(y,np.arange(0,1.01,.2),1)
f2=save("figure2a_age99",centers(ds,11,325),xf,yf,np.arange(20,301,5),"cash_on_hand_thousand_usd")
print("Figure2A",[v for v in f2 if v[0]%25==0])
print("Figure2A complete market line",float(np.polyval(yf,ds[325]["items"][0][1].y)),"theory",.04/(10*.157**2))

p=doc[14];ds=p.get_drawings()
# The panel B tick marks lie halfway between labelled ages: use the numeric
# labels' centers, not a false assumption that the first tick is age 20.
words=[w for w in p.get_text("words") if 349<w[1]<352 and w[4].isdigit()]
x=np.array([.5*(w[0]+w[2]) for w in words]);xvalues=np.array([float(w[4]) for w in words])
assert len(x)==16
xf=np.polyfit(x,xvalues,1)
y=np.array([it[1].y for it in ds[329]["items"][1:9]])
yf=np.polyfit(y,np.arange(0,701,100),1);yrf=np.polyfit(y,np.arange(0,36,5),1)
points=[]
for idx,items in [(329,ds[329]["items"][27:]),(330,ds[330]["items"][:35])]:
 for it in items:
  points.extend([[q.x,q.y,idx] for q in it[1:]])
f3p=save("figure3b_npv_income",points,xf,yf,np.arange(20,100),"age_as_printed")
f3r=save("figure3b_npv_to_wealth",centers(ds,331,525),xf,yrf,np.arange(20,100),"age_as_printed")
print("Figure3B",[(*a,b[1]) for a,b in zip(f3p,f3r) if a[0]%5==0])
meta={"figure2a":{"page_1_based":12,"printed_page":502,"drawing_indices_age99":[11,324],"y_top_border_value":float(np.polyval(yf,56.091)),"x_affine":None,"warning":"Figure2A x is cash-on-hand (thousands of 1992 USD), not savings."},"figure3b":{"page_1_based":15,"printed_page":505,"x_label_pdf_pt":x.tolist(),"x_label_values":xvalues.tolist(),"x_affine":xf.tolist(),"y_affine":yf.tolist(),"ratio_y_affine":yrf.tolist(),"first_curve_age_as_printed":float(np.polyval(xf,points[0][0])),"x_axis_anomaly":"Age labels are offset by half a 5-year tick interval. Curves start near printed age 18 and end near 98. Do not force first curve point or first tick to age 20.","units":"NPV in thousands of 1992 USD; ratio dimensionless"}}
# Re-read Fig2 transforms here to avoid using the overwritten Fig3 transform.
it=doc[11].get_drawings()[10]["items"]
f2xf=np.polyfit([q[1].x for q in it[8:]],np.arange(0,301,25),1)
f2yf=np.polyfit([q[1].y for q in it[1:7]],np.arange(0,1.01,.2),1)
meta["figure2a"].update(x_affine=f2xf.tolist(),y_affine=f2yf.tolist(),y_top_border_value=float(np.polyval(f2yf,56.091)))
(OUT/"additional_extraction_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")

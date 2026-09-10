from pathlib import Path
import os,csv
out=Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"]=str(out/".matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read(name):
 rows=list(csv.DictReader((out/name).open(encoding="utf-8")))
 key=list(rows[0])[0]
 return [float(r[key]) for r in rows],[float(r["value"]) for r in rows]
fig,ax=plt.subplots(2,1,figsize=(8,7),layout="constrained")
x,y=read("figure2a_age99_raw.csv");ax[0].plot(x,y,label="Age 99")
ax[0].axhline(.04/(10*.157**2),color="gray",label="Complete markets (theory)")
ax[0].set(xlabel="Cash-on-hand, thousands of 1992 USD",ylabel="Stock share",xlim=(0,300),ylim=(0,1.1),title="CGM Figure 2A: extracted vector curve")
x,y=read("figure3b_npv_income_raw.csv");ax[1].plot(x,y,label="NPV(Y)")
b=ax[1].twinx();x,y=read("figure3b_npv_to_wealth_raw.csv");b.plot(x,y,ls="--",color="orange",label="NPV(Y)/W as printed")
ax[1].set(xlabel="Age (printed labels)",ylabel="Thousands of 1992 USD",xlim=(18,99),ylim=(0,700),title="CGM Figure 3B: aggregation and age convention unconfirmed")
b.set(ylabel="Ratio",ylim=(0,35))
for a in ax:a.grid(alpha=.3);a.legend(loc="upper left")
b.legend(loc="upper right")
fig.savefig(out/"additional_extracted.png",dpi=150)

import sys, os, json, time
import numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE,"..","discrete"))
import nn_lifecycle as L
from cgm_model import solve as solve_grid, simulate as simulate_grid

t0=time.time()
net,hist,secs=L.train(iterations=2500, batch=192, nalpha=41, verbose=True)
print("trained in %.0f s"%secs)
torch.save(net.state_dict(), os.path.join(HERE,"..","results","nn_lifecycle.pt"))
print("extracting policy ...")
t1=time.time()
nnpol=L.extract_policy(net, na=400, amax=80.0, nalpha=201)
print("policy extracted in %.0f s"%(time.time()-t1))
np.savez(os.path.join(HERE,"..","results","nn_lifecycle_policy.npz"),
   **{"x_%d"%i:nnpol["x"][i] for i in range(len(L.AGES))},
   **{"c_%d"%i:nnpol["c"][i] for i in range(len(L.AGES))},
   **{"a_%d"%i:nnpol["alpha"][i] for i in range(len(L.AGES))})
# simulate with the NN policy using the repo simulator
sim=L.simulate_grid(nnpol, N=20000, seed=7)
ages=np.asarray(nnpol["ages"])
i65=int(np.where(ages==65)[0][0])
print("NN policy simulation: wealth peak %.2f at age %d ; alpha@65 %.4f ; C@65 %.2f"%(
  sim["meanW"].max(), ages[int(np.argmax(sim["meanW"]))], sim["meanA"][i65], sim["meanC"][i65]))
json.dump(dict(train_seconds=secs, iterations=2500, history=hist,
               wealth_peak=float(sim["meanW"].max()),
               peak_age=int(ages[int(np.argmax(sim["meanW"]))]),
               alpha65=float(sim["meanA"][i65]), C65=float(sim["meanC"][i65])),
          open(os.path.join(HERE,"..","results","nn_lifecycle_training.json"),"w"), indent=2)
print("done %.0f s total"%(time.time()-t0))

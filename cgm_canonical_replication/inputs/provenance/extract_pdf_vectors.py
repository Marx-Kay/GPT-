"""Optional input-provenance tool, NOT used by MATLAB run_all.
Requires PyMuPDF solely to re-extract PDF drawing primitives. Canonical axes,
curve identities, interpolation and ALL comparisons are regenerated in MATLAB.
"""
from pathlib import Path
import json
import pymupdf
root = Path(__file__).resolve().parents[2]
pdf = pymupdf.open(root / "inputs/paper/cgm.pdf")
output = {}
for page in [11, 14, 32]:
    drawings = []
    for d in pdf[page].get_drawings():
        points = []
        for item in d["items"]:
            if item[0] == "l":
                points += [[item[1].x, item[1].y], [item[2].x, item[2].y]]
            elif item[0] == "re":
                q = item[1]
                points += [[q.x0,q.y0], [q.x0,q.y1], [q.x1,q.y0], [q.x1,q.y1]]
            else:
                raise ValueError("Unexpected primitive: " + str(item[0]))
        drawings.append(dict(width=d["width"], kind=d["type"], rect=list(d["rect"]), points=points))
    output["page" + str(page + 1)] = drawings
(root / "inputs/paper_targets/pdf_vector_paths.json").write_text(json.dumps(output), encoding="utf-8")

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Plot only the original saved training predictions; do not train a model."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

site = Path(__file__).resolve().parents[1]
data = json.loads((site / "src/data/archive.json").read_text())
rows = range(1, 5)
plt.rcParams.update({"font.family": "serif", "font.size": 12})
fig, ax = plt.subplots(figsize=(7.6, 3.4), layout="constrained")
ax.hlines(rows, data["targets"], data["predictions"], color="black", linewidth=1)
ax.scatter(data["targets"], rows, marker="o", facecolors="white",
           edgecolors="black", s=55, label="Observed", zorder=3)
ax.scatter(data["predictions"], rows, marker="x", color="black",
           s=55, label="Predicted", zorder=4)
ax.set(xlim=(0, 100), ylim=(0.5, 4.5), xlabel="Score out of 100")
ax.set_yticks(list(rows), [f"Row {row}" for row in rows])
ax.set_xticks([0, 25, 50, 75, 100])
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="0.85", linewidth=0.5)
ax.legend(loc="lower left", frameon=False)
fig.savefig(site / "src/assets/predictions.png", dpi=160)
plt.close(fig)

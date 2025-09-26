# Liquid Neural Morphogenesis in Cellular Automata



<img width="843" height="759" alt="Screenshot 2025-09-26 at 15 47 41" src="https://github.com/user-attachments/assets/f0b65b98-4755-4b52-8300-8471b53486ae" />



A liquid‑neural cellular automaton (LN‑CA) that **grows** target morphologies from a single seed and **self‑heals** after damage, trained and evaluated on an NVIDIA **A100‑40GB** in PyTorch 2.8 / CUDA 12.6. 

---

## What's inside

* **Liquid‑Neural CA rule** learned end‑to‑end; updates are *local* and *homogeneous* across the grid.
* **Growth & Regeneration curriculum** (Stage‑A/B).
* **Generalization** to **size/rotation** via a steering potential $\phi(x,y)$.
* **Environment‑steered morphogenesis** with a **diffusing chemical field**.
* **ASAL‑style novelty search** over environments to discover new emergent morphologies.

**Observed outcomes:**

* Stage‑A (growth) converges to ~**0.041** total loss (example run). 
* Stage‑B (regeneration) recovers after mid‑rollout damage, ~**0.062** loss. 
* Model size: **~0.0084 M** parameters. 

See the notebook cells and screenshots for qualitative results (growth from a seed to **ring / diamond / bi‑lobe**; and robust **healing** after damage). Examples appear around **pages 12–16** of the exported notebook PDF. 

---

## Problem & Approach

We want **global morphologies** (thin rings, diamonds, bi‑lobes) to **emerge from purely local rules**, *and* those rules should be **adaptive**—capable of **self‑repair** without changing parameters at test time.

We instantiate a **neural cellular automaton** whose per‑cell update is a **Liquid Neural Network (LNN)** unit with an **input‑dependent gate** (a continuous‑time–like update), plus an **alive‑gated locality mask** so only cells near living tissue update strongly. This yields a **moving growth front** that stops on target contours and restarts when tissue is damaged.

---

## Model: Liquid‑Neural CA (LN‑CA)

### State layout (per cell)

* **Visible (logits)**: `RGB` (3), `ALIVE` (1) — we keep **logits** and apply `sigmoid` only for visualization/losses.
* **Hidden memory**: `HIDDEN = 16` liquid units per cell.
* **Instruction**: one‑hot (3 channels) for ring / diamond / bi‑lobe.
* **Spare**: 1 channel (for seed mark **or** steering fields like $\phi$ or the chemical).
  Default grid: **64×64**. See `Config` on pages 2–3. 

### Neighborhood & updates

Each step, a cell reads a **3×3** neighborhood over all channels (unfold), then:

1. **Liquid hidden update**

$$
\tilde{h}=\tanh(W_x n + W_h h),\quad
\Delta=\sigma(W_g[n,h])\in[\delta_{\min},\,1-\delta_{\min}],
$$

$$
h^{+} = h + m \odot \Delta \odot (\tilde{h}-h),
$$

with **$\Delta$** the *input‑dependent time constant* (the "liquid" gate), and **$m$** a **local update mask** from ALIVE (below).

2. **Visible logits update** (with tiny **logit leak** to avoid saturation):

$$
[\Delta \mathbf{y}, \Delta a] = W_o[h^{+},n],\quad
\mathbf{y}^{+}=(1-\lambda)\mathbf{y}+m\odot\Delta \mathbf{y},\quad
a^{+}=a+m\odot\Delta a.
$$

3. **Local update mask** from ALIVE:

$$
m=\sigma\Big(\kappa\,[\text{AvgPool}_{3\times 3}(\sigma(a))-\tau]\Big),
$$

so only **near‑alive** cells update strongly → a **bounded, advancing front**.

Module initialization, gating, leak, and neighborhood wiring are shown in **Cell 4** of the notebook. 

---

## Training curriculum

### Stage‑A: **Growth**

Seed a **3×3** alive patch at a random location; generate a target (ring / diamond / bi‑lobe) *centered on the seed*. Rollouts are 48–96 steps (randomized). 

### Stage‑B: **Regeneration**

Same setup, but we **damage** a random rectangle mid‑rollout by driving visible logits strongly negative; the rule must **repair** the missing tissue. 

### Losses (applied to final step + **mid‑step** at 0.75T)

* **Foreground‑weighted RGB MSE** (thin edges matter).
* **Total Variation** (contour smoothness).
* **BCE on ALIVE logits** vs. target mask (with `pos_weight` for class imbalance).
* Small regularizers: mean ALIVE and hidden L2.
  Weights and mid‑step mixing appear in **Cells 2 & 5**. 

**Why this works:** the **ALIVE BCE** teaches *where* the front should live; the **foreground MSE + TV** shape *how* it should look; **mid‑step supervision** stabilizes dynamics; **local gating** prevents the "paint the canvas" failure.

---

## Results

* **Growth** reaches **~0.0409** total loss (example printout), with visually crisp contours by step 63. See **page 12**. 
* **Regeneration** reaches **~0.0615**, with the front re‑entering the damaged region to repair the ring/diamond/bi‑lobe. See **pages 14–15**. 
* **Loss curve** shows steady log‑scale descent across Stage‑A and Stage‑B (page 17). 
* **Fixed‑gate ablation** (no liquid adaptation) diffuses and fails to form/repair clean contours (page 18). 

---

## Generalization & Environment Extensions

### (a) **Size/Rotation generalization** with $\phi(x,y)$

We synthesize a **steering potential** whose minimum lies on the target contour and place it in the **SPARE** channel; sizes in **75–125%** and rotations **±45°** are randomized during training. The same rule draws the desired shape across these variations. See **Cells 14–16**, training logs on **page 30**. 

### (b) **Diffusing chemical field** (dynamic environment)

We add a **2‑D chemical PDE** (discrete Laplacian diffusion + decay + Gaussian sources). The CA reads the chemical in SPARE and learns to "wrap" the **super‑threshold** region; this yields environment‑steered morphogenesis that adapts as the field evolves. See **Cells 17–19** and examples on **page 30**. 

### (c) **ASAL‑style novelty search**

We randomly sample chemical environments (diffusion, decay, sources, threshold), summarize the final morphology by a 6‑D descriptor (**area, perimeter, circularity, centroid, anisotropy**), and rank by **novelty** (mean distance to nearest archive). We then **fine‑tune** on the most novel winners as a curriculum booster. See **Cells 20–22**, outputs on **pages 31–32**. 

---

## Quickstart

### Colab

Open the notebook and **Run all** cells. The environment banners confirm **PyTorch 2.8.0 + CUDA 12.6** and A100 GPU (see **page 1** of the PDF). 

### Local (optional)

```bash
conda create -n lnca python=3.10 -y
conda activate lnca
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install numpy matplotlib
# then run the notebook
```

---

## Key configuration (defaults)

* Grid: `H=W=64`
* Channels: `RGB(3) + ALIVE(1)` (logits), `HIDDEN=16`, `INSTR=3`, `SPARE=1`
* Liquid gate floor: `DELTA_MIN=0.02`
* Local update mask: `ALIVE_THRESH=0.10`, `ALIVE_KAPPA=10.0`
* Logit leak: `LOGIT_LEAK=0.01`
* Rollout length: `T∈[48,96]`, mid‑step weight `0.30`
* Loss weights: `MSE=1.0`, `TV=0.02`, `ALIVE_BCE=0.7`, `alive_mean=1e‑4`, `hidden_L2=1e‑3`
* Foreground emphasis: `FG_WEIGHT=8.0`
  See `Config` printing around **pages 2–3**. 

---

## Reproducing

1. **Train Stage‑A** (growth) → visual checks (page 12–13). 
2. **Train Stage‑B** (regen) → damage mid‑rollout (page 15). 
3. (Optional) **Generalization**: run Cell 23 (page 30). 
4. (Optional) **Chemical steering**: run Cell 24 (page 30). 
5. (Optional) **ASAL search & curriculum**: run Cell 27 (pages 31–32). 

---

## Ablations & Diagnostics

* **Fixed‑gate vs Liquid‑gate:** fixed gate shows diffuse artifacts and weak healing (page 18). Liquid gating stabilizes the growth front and improves repair. 
* **Common pitfalls & fixes**

  * **Canvas washout** → raise `ALIVE_THRESH` a bit (e.g., 0.12–0.15) or increase `FG_WEIGHT`.
  * **Growth stalls** → increase `C_HIDDEN` (e.g., 20) or `self.Wout` scale slightly.
  * **Ragged edges** → increase `TV` or lower `LOGIT_LEAK` from 0.01→0.005.

---

## License & Citation

* **License:** MIT (recommendation; adjust to your preference).
* **Citation (project README)**

  > *Alfaxad Eyembe, "Liquid Neural Morphogenesis in Cellular Automata," 2025, code & notebook.*

For precise reproducibility and figures referenced above, see the included **exported notebook PDF** (page references in this README correspond to that file). 

---

## Acknowledgments

Inspired by prior work on **Neural Cellular Automata** and **Liquid Neural Networks**; this project blends liquid gating with alive‑gated locality to achieve robust morphogenesis and self‑healing in CA form.

---

# Fly-brain Worldle

Can a fruit fly's brain learn to play [Worldle](https://worldle.teuteuf.fr/)?

This project wires two pieces of real *Drosophila* connectome data into a reinforcement-learning agent and trains it
to guess a country from its silhouette, then home in using the game's direction + distance feedback:

- **Vision:** [flyvis](https://github.com/TuragaLab/flyvis), a pretrained, connectome-constrained model of the fly
  optic lobe (65 cell types, 45,669 cells, R1–R8 → lamina → medulla → T4/T5). It is frozen and used as the agent's eyes.
- **Decision:** a recurrent network whose wiring is the **central complex** of the
  [MaleCNS connectome](https://male-cns.janelia.org/) (Janelia FlyEM, v1.0), aggregated to cell types. The central
  complex is the fly's real heading-integration circuit, which is the biological job this game demands: combine
  "the answer is 3,000 km north-west of your last guess" across several guesses.

Only synaptic gains on *existing* connections, per-type biases and time constants are trained; the graph and the
excitatory/inhibitory signs (from predicted neurotransmitters) are fixed by the connectome.

![demo](docs/demo.png)

## How it works

```
country silhouette
      │  rendered as a short drifting movie (the fly visual system is motion-driven)
      ▼
flyvis optic lobe  (frozen)             ──►  T4/T5 responses, retinotopic, 5,768 features
      │
      ▼
central-complex RNN  ◄──  last guess's feedback: compass direction (8-way, as in Worldle), distance, location
      │  topology + signs from MaleCNS; gains trained
      ▼
policy head  ──►  (heading, distance) from the previous guess, Gaussian
      │
      ▼
destination point on the globe  ──►  nearest country  ──►  guess
```

The game is re-implemented locally ([`flybrain_worldle/game`](flybrain_worldle/game)) from public-domain
[Natural Earth](https://www.naturalearthdata.com/) 1:10m boundaries, so the agent can play unlimited episodes with
any of 195 countries as the answer: the 193 UN member states plus Vatican City and Palestine
(`scripts/build_countries.py`). Northern Cyprus is drawn as part of Cyprus and Somaliland as part of Somalia;
Western Sahara, Kosovo and Taiwan are not UN members and are not in the game. Overseas territories are dropped so
the silhouette is the shape people recognise. Feedback follows Worldle: great-circle distance in km, one of eight
compass directions, and a proximity percentage. Training is REINFORCE with a moving baseline
([`training/reinforce.py`](flybrain_worldle/training/reinforce.py)).

## Results

Greedy policy, every one of the 195 countries as the answer once, 4,000 REINFORCE iterations × 128 episodes, one seed:

| vision | central-complex graph | solved ≤ 6 | mean guesses | 1st-guess accuracy |
|---|---|---|---|---|
| **flyvis (fly optic lobe)** | **MaleCNS wiring** | **96%** | **3.61** | **3%** |
| flyvis | MaleCNS, shuffled | 95% | 3.27 | 5% |
| flyvis | random sparse stub (32 types) | 95% | 3.23 | 3% |
| raw 16×16 pixels | MaleCNS wiring | 99% | 3.07 | 8% |

Chance first-guess accuracy is 0.5%.

![learning curves](docs/learning_curves.png)

### What I take from this

The fly is good at the navigation half of the game and bad at the recognition half.

Once it has one piece of feedback, it plays roughly like a person would: "3,000 km north-west of Saudi Arabia"
gets turned into a sensible next guess, and it lands on the answer in three or four tries almost every time. That
part is the central-complex RNN integrating headings, and it works whether the network is wired like the real
central complex, wired at random with the same synapses, or is a small random graph. So I can't say the real wiring
helps. 96% against 95% and 95% is a single-seed coin flip, and I'd want several seeds and a harder task before
reading anything into it.

The first guess is where it struggles. Three percent is about six times chance, but a plain 16×16 pixel
downsample does better (8%), which tells me the frozen optic lobe isn't a great shape detector. That's not really a
surprise: T4/T5 cells evolved to see motion, the model was trained to estimate optic flow, and I never fine-tuned it
on silhouettes. Going from 168 to 195 countries made this worse for every variant, because a lot of the additions
are small island states whose outlines are near-identical blobs at this resolution. If I were to push the project
further, the eyes are the thing to work on, not the compass.

One practical lesson: with the full 598-type graph, plain REINFORCE climbed to about 80% and then collapsed to 10%
mid-run until I put a floor under the policy's standard deviation. The small 32-type stub never did this, which
is a good argument for running the controls early rather than at the end.

Trained checkpoints and logs for the four runs above are in [`runs/`](runs/), and the flyvis features for every
country are cached in `data/cache/`, so the demo and evaluation work from a clone without the pretrained
optic-lobe download.

Controls: `--vision pixels` replaces the fly optic lobe with a raw 16×16 downsample; `--graph shuffled` keeps the
central-complex edge weights but rewires them at random; `--graph stub` is a random sparse graph.

## Run it

```bash
conda create -n flybrain python=3.11 && conda activate flybrain
pip install -e .[dev]
flyvis download-pretrained          # pretrained optic-lobe ensemble (~once)
pytest                              # 30 tests, no network needed

# 1. central-complex connectivity from neuPrint (free account: https://neuprint.janelia.org/account)
export NEUPRINT_TOKEN=...
python scripts/fetch_connectome.py  # -> data/cx_types.csv, data/cx_edges.csv

# 2. train (first run renders every country through flyvis once, ~10 min on CPU, then cached)
python scripts/train.py --vision flyvis --graph cx
python scripts/train.py --vision pixels --graph cx        # control: no fly vision
python scripts/train.py --vision flyvis --graph shuffled  # control: no fly wiring

# 3. evaluate and plot
python scripts/evaluate.py runs/*/best.pt
python scripts/plot_runs.py runs/flyvis_cx_s0 runs/pixels_cx_s0 runs/flyvis_shuffled_s0

# 4. (optional, needs NEUPRINT_TOKEN) brain geometry for the 3D view: neuropil meshes + one skeleton per cell type
python scripts/fetch_brain_geometry.py   # -> data/brain/brain.json.gz (committed, so a clone already has it)

# 5. watch it play
python scripts/run_demo.py --checkpoint runs/flyvis_cx_s0/best.pt   # http://127.0.0.1:8000
```

The demo shows the silhouette, the guesses with Worldle's feedback (the km shown is the distance from *that guess*
to the hidden answer), a map that zooms to the area in play and highlights each guessed country (the yellow
squares are the raw (heading, distance) proposals before snapping to the nearest country), and a 3D central
complex: one representative neuron per cell type from MaleCNS, drawn inside the EB/PB/FB/NO neuropil meshes and
coloured by the RNN's activity at each guess.
The 3D view is plain Three.js on the exported geometry; for real analysis use
[navis](https://navis-org.github.io/navis/) (Python, talks to neuPrint directly) or
[neuroglancer](https://github.com/google/neuroglancer), which is what neuPrint and FlyWire use.

## Honest scope

- flyvis's connectome is the FIB-25/FIB-19 optic-lobe reconstruction, not MaleCNS. Only the central-complex RNN is
  built from MaleCNS data. Both are real fly connectomes; the README does not claim more than that.
- The central complex is used at **cell-type** resolution (a few hundred types), not single neurons, and its
  inputs/outputs are trainable linear projections rather than the real sensory pathways into the CX.
- Neuron dynamics are a leaky rate model, not spiking or biophysical.
- The visual features are frozen, so the "fly vision" contribution is exactly what the pretrained optic lobe already
  computes; nothing in the optic lobe is tuned for this task.

## Layout

```
flybrain_worldle/
  game/         countries, silhouettes, great-circle geometry, WorldleEnv
  connectome/   central_complex.py (neuPrint fetch + RNN), vision.py (flyvis wrapper), agent.py, anatomy.py (meshes + skeletons)
  training/     REINFORCE loop, batched torch environment, config
  demo/         FastAPI backend + single-file frontend
scripts/        build_countries, fetch_connectome, fetch_brain_geometry, train, evaluate, plot_runs, run_demo
tests/          geometry, environment, agent/graph shape tests
```

## Data and citations

- MaleCNS v1.0 connectome, Janelia FlyEM, CC-BY: https://male-cns.janelia.org/
- Lappalainen et al., "Connectome-constrained networks predict neural activity across the fly visual system",
  *Nature* (2024). https://doi.org/10.1038/s41586-024-07939-3 — flyvis, MIT license.
- Nern et al., "Connectome-driven neural inventory of a complete visual system", *Nature* (2025).
- Natural Earth 1:10m cultural vectors, public domain.
- Worldle by teuteuf; this repository re-implements the rules and does not touch the site.

## Acknowledgments

Built with the assistance of [Claude Code](https://claude.com/claude-code).

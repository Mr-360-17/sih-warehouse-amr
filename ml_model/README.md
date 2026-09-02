# Demand Prediction (Edge-AI layer)

**PS 26123 — Edge-AI Based Distributed Fleet Coordination for Autonomous Mobile Robots (AMRs) in Smart Warehouses**
Team 404: Human Not Found

This is the layer that moves the fleet from **reactive** to **predictive**. Given the hour of day, it predicts which warehouse section (A/B/C/D) the next order is most likely to come from, so an idle AMR can pre-position toward that section instead of waiting at its dock for a task that has not been assigned yet.

## Files

| File | What it does |
|---|---|
| `generate_data.py` | Generates the synthetic order log (`../warehouse_orders.csv`). Seeded, so it is reproducible. |
| `train_mlp.py` | Loads the data, trains a scikit-learn `MLPClassifier`, validates it on held-out data, prints a full report. |
| `export_weights.py` | Exports the trained weights to JSON + JavaScript so the browser simulation runs the *same* model. |
| `artifacts/` | Generated outputs — model, weights, metrics. |

## Running it

```bash
python ml_model/generate_data.py     # writes warehouse_orders.csv (1000 orders)
python ml_model/train_mlp.py         # trains + validates, writes artifacts/
python ml_model/export_weights.py    # writes artifacts/mlp_weights.json + predict_zone.js
```

## The model

```
24 one-hot hour inputs  ->  6 ReLU hidden units  ->  4 softmax outputs (A/B/C/D)
```

`sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(6,), activation='relu')`, 80/20 stratified split, `random_state=42` throughout.

**Why one-hot and not the raw hour.** Hour 23 and hour 0 are adjacent in real time but maximally far apart as integers. Feeding `hour` in as a plain number forces the network to learn a jagged, discontinuous function from a single input. One-hot encoding removes that false ordering — each hour becomes its own independent switch, which matches how the pattern was actually generated.

## Results

Held-out test set (200 rows), seeded run:

| | |
|---|---|
| random guessing over 4 sections | 25.0% |
| **our model** | **43.0%** |
| theoretical maximum | 42.5% |
| **on the 8 busy hours** | **75.7%** |
| on the 16 quiet hours | 23.8% |

All eight planted busy hours are predicted correctly:

```
08:00 → A    09:00 → A    10:00 → A    11:00 → A
17:00 → B    18:00 → B    19:00 → B    20:00 → B
```

## Read the accuracy correctly

**43% is not a weak result — it is very close to the best score that is mathematically possible on this data.**

`generate_data.py` deliberately plants a pattern in only part of the day. Section A is busy 08–11, Section B is busy 17–20; sections C and D are never given a busy window. During a busy hour, 70% of orders come from that section and 30% are spread evenly across all four. During the other **16 hours the section is drawn uniformly at random — there is no signal to learn.**

So the ceiling is:

```
busy  hours:   8/24 × (0.70 + 0.30 × 0.25)  =  8/24 × 0.775
quiet hours:  16/24 × 0.25
                                     total  ≈  42.5%
```

Two consequences worth knowing before anyone asks:

- **A score near 42–43% means the model found essentially everything there is to find.** A score near 90% would mean something is broken — data leakage, or evaluating on the training set.
- **Scoring marginally *above* 42.5% is normal.** The ceiling is an expected value over infinite data; on a 200-row test set the score lands either side of it by chance. `train_mlp.py` prints a note when this happens rather than claiming ">100% of signal recovered".

The honest headline is therefore **not** "43% accuracy". It is:

> Random guessing is 25%. The maximum achievable on this data is 42.5%, because we deliberately made two-thirds of the day pure noise. Our model reaches 43% — at the ceiling — and on the busy hours it actually cares about it is 75.7% accurate, predicting all eight correctly.

## Integration with the simulation

The warehouse simulation is a single dependency-free HTML file and cannot import scikit-learn. Rather than hand-writing a second, different network in JavaScript and hoping it behaves the same, `export_weights.py` dumps the **actual trained weights** and generates `artifacts/predict_zone.js`:

```js
predictZone(9)   // → { section: "A", scores: { A: 0.803, B: 0.029, C: 0.065, D: 0.103 } }
predictZone(18)  // → { section: "B", scores: { A: 0.039, B: 0.716, C: 0.116, D: 0.130 } }
```

The exporter verifies that the JavaScript forward pass reproduces scikit-learn's prediction for **all 24 hours** before writing the file, and fails loudly if it does not.

This is what lets the deck say "scikit-learn MLPClassifier" and have it be literally true of what runs in the demo — the browser is executing the same weights, trained and validated here in Python.

## Next step

Wire `predictZone()` into the simulation so idle AMRs pre-position toward the predicted section. The scores it returns are already in the shape the fleet dashboard's hot-zone panel displays.

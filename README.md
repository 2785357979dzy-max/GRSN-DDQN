# GRSN-DDQN

A Graph-based Road Scoring Network (GRSN) combined with a Double Deep Q-Network (DDQN) for post-disaster road recovery sequencing.

## Installation

Python 3.12 and a dedicated virtual environment are recommended. For CPU execution:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For GPU execution, install the appropriate version following the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/#v271) and use `--device cuda` when running the commands below.

## Input

Prepare NumPy `.npz` files and specify their paths following [manifest.example.json](manifest.example.json). Relative paths are resolved from the directory containing the manifest.

The input network must be a simple undirected graph with the following arrays, where N is the number of nodes and K is the number of roads:

| Array | Shape | Description |
|---|---|---|
| `endpoints` | `(2, K)` | Integer endpoint indices from 0 to N−1; list each undirected road once, without self-loops |
| `length_m` | `(K,)` | Road length in metres |
| `speed_mps` | `(K,)` | Speed under undamaged conditions, in metres per second |
| `population` | `(N,)` | Population assigned to each node |
| `importance` | `(K,)` | Fraction of shortest travel-time paths in the undamaged network that pass through each road, ranging from 0 to 1 |

Each scenario file contains a `damage` array of shape `(K,)`, ordered consistently with the roads: 0 indicates undamaged, 0.5 partially blocked, and 1 fully blocked. Road lengths and undamaged speeds must be positive. Node populations must be nonnegative, with a positive total. Partially blocked roads operate at half their undamaged speed; fully blocked roads are impassable. Repair duration is `damage × length_m / 500`, expressed in model repair-time units.

## Usage

Run the following commands from the repository root, replacing the manifest path. Training requires nine training scenarios:

```bash
python -m grsn_ddqn.train --manifest /path/to/manifest.json --config configs/paper.json --output outputs/run_01 --device cpu
```

Generate a recovery sequence:

```bash
python -m grsn_ddqn.predict --manifest /path/to/manifest.json --scenario test_1 --checkpoint outputs/run_01/model.pt --output outputs/test_1.json --device cpu
```

## Data Availability

**The study dataset is not shared due to confidentiality restrictions.** Related publicly available source materials can be obtained through the following channels.

| Material | Source |
|---|---|
| Road network | [OpenStreetMap](https://www.openstreetmap.org/) / [Geofabrik China extracts](https://download.geofabrik.de/asia/china.html), subject to the [OSM licence](https://www.openstreetmap.org/copyright) |
| 100 m population grid | [Figshare dataset](https://doi.org/10.6084/m9.figshare.24916140.v1); [data paper](https://doi.org/10.5194/essd-16-3705-2024) |
| Disaster records | [National Earth System Science Data Center](https://www.geodata.cn/data/datadetails.html?dataguid=159979431735834&docid=5235) |
| Henan rainstorm information | [ArcGIS thematic map](https://storymaps.arcgis.com/stories/bf86150b748e4dbbb3a4528625f841e7) |
| Local reports | [Zhengzhou Bendibao](https://zz.bendibao.com/) and other public news sources |

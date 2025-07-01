# CausalIQ: Causal Discovery Research

> ⚠️ **This repository is under active restructuring. Expect significant changes.**

This repository hosts the code and reproducibility infrastructure behind a series of academic papers on **Bayesian Network structure learning**, developed by [Ken Kitson](https://github.com/KenKitson) and collaborators.

It includes:
- New **structure learning algorithms** (e.g., Tabu-Stable)
- Tools for evaluating **accuracy** and **stability**
- Integration of **human expertise** in causal discovery
- A framework for **reproducing published figures and results**

---

## 📦 Project Goals

- Provide an installable Python package of reusable algorithms (`causaliq/`)
- Provide an installable Python package implementing a pipleine (`causaliq_pipeline/`) to run causaliq and other structure learning algorithms and analyse the results
- Offer a reproducibility framework for datasets, experiments, figures, and tables from papers which downloads large files and other assets from Zenodo

---

## 🔁 Reproducibility Workflow (Coming Soon)

You can reproduce figures and tables from supported papers using the `reproduce.py` script with the following arguments:

- `--paper` short name for the paper (e.g. `2025kitsoncausal`) which matches the bib citation label
- `--mode` controls how much is reproduced from scratch:
  - `--learn` run structure learning from datasets and perform analysis
  - `--analyse` perform analysis using structure learning trace files downloaded from Zenodo to generate the paper tables and figures (avoiding structure learning)
  - `--download` just download tables and figures from Zenodo
- `--overwrite` whether to regenerate files already on systems (otherwise make use of datasets, trace files, tables and figures already present locally)

**Example:**

```
python reproduce.py --paper kitson2025stable --mode learn --item figure4
```


## 📁 Planned Directory Structure

```
causaliq/          # Python package: algorithms to run on new data
causaliq_pipeline/ # Python package: learning & analysis pipelines
tests/             # Unit tests
repro/             # Reproducubility assets e.g. data, traces, figures
README.md
setup.py
```

---

## 📚 Related Papers

This repository **will soon** support experiments from, for example:

- `kitson2025causal` Kitson, N.K. and Constantinou, A.C., 2025. Causal discovery using dynamically requested knowledge. _Knowledge-Based Systems_, 314, p.113185.
- `kitson2024impact` Kitson, N.K. and Constantinou, A.C., 2024. The impact of variable ordering on Bayesian network structure learning. _Data Mining and Knowledge Discovery_, 38(4), pp.2545-2569. [https://doi.org/10.1007/s10618-024-01044-9](https://doi.org/10.1007/s10618-024-01044-9)


More papers will be added soon.

---


## 🔧 Using the Algorithms (planned)

Once packaged, install with:

```bash
pip install causaliq
```

Use in your own code (COMING SOON), this is a possble example:

```python
from causaliq.learning import tabustable
G = tabustable.learn_structure(df, score="bic", tabu_length=15)
```

---


---

## 📜 License

This project is licensed under the MIT License.

---

## ✉️ Contact

For issues or collaboration, please open a GitHub issue or contact [Ken Kitson](https://github.com/KenKitson).

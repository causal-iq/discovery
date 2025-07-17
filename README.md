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

You can reproduce figures and tables from supported papers using the `repro.py` script with the following arguments:

```bash
python repro.py <operation> <target> [--run]
```

**Arguments:**

- `<operation>` (positional): defines what te script will do, one of:
  - `learn` - performs all required structure learnng and analysis to generate asset
  - `analyse` - downloads structure learning results but performs analysis to generate analysis
  - `download` - downloads required results and asset
- `<target>` (positional): Path to the asset to reproduce (e.g., `papers/ijar_stable/fig1`)
- `--run`: Actually perform the operation (omit for a dry run with time estimates)
- `--help`: Show usage instructions

**Examples:**

- Dry run (shows what structure learning and analysis would be done to reproduce fig1 of paper ijar_stable):  
  `python repro.py learn papers/ijar_stable/fig1`
- Actually download structure learning results, but rerun analysis locally for tab2:  
  `python repro.py analyse papers/ijar_stable/tab2 --run`
- Just download results and selected assets from Zenodo:  
  `python repro.py download papers/ijar_stable/fig1`

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

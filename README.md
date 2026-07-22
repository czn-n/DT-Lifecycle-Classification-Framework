# Life Cycle-Oriented Digital Twin for Civil Infrastructure Assets: A natural language processing model-assisted review

This repository is associated with the manuscript:

"Life Cycle-Oriented Digital Twin for Civil Infrastructure Assets: A natural language processing model-assisted review"

## Overview

The classification pipeline utilizes a hybrid NLP approach combining **rule-based multi-tier keyword matching** and **Sentence-BERT (SBERT) semantic vector similarity** to automatically assign civil infrastructure Digital Twin literature records into four key lifecycle stages:
1. **Design & Planning** (`Design_Planning`)
2. **Construction** (`Construction`)
3. **Operation & Maintenance** (`Operation_Maintenance`)
4. **Demolition & Renovation** (`Demolition_Renovation`)

It also includes an **uncertainty screening mechanism** to flag low-confidence predictions that require manual human-in-the-loop verification.

---

## Environment & Installation

Make sure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install pandas numpy sentence-transformers
```
## Usage

Run the classifier using your literature metadata CSV file download from Scopus dataset. The input file should contain columns such as **Title**, **Abstract**, **Author Keywords**, and **Index Keywords**.

```bash
python lifecycle_classification.py --input_csv your_literature_data.csv --output_dir ./output --low_conf_thresh 0.25
```

---

## Citation

If you find this code, framework, or methodology useful in your research, please cite our paper:

```bibtex
@article{chen2026life,
  title={Life Cycle-Oriented Digital Twin for Civil Infrastructure Assets: A natural language processing model-assisted review},
  author={Chen, Ziyang and Wang, Jiehui and Ueda, Tamon and Dai, Jian-Guo},
  journal={Journal of Infrastructure Intelligence and Resilience},
  pages={100225},
  year={2026},
  publisher={Elsevier}
}
```

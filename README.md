# 🧠 Simulated Annealing Portfolio Optimizer

This is a Streamlit app that solves a **portfolio optimization problem** using **classical simulated annealing** instead of quantum annealing.

---

## 📈 What It Does

- Lets you adjust **risk aversion**
- Translates portfolio optimization into a **QUBO** (Quadratic Unconstrained Binary Optimization) problem
- Solves it with `SimulatedAnnealingSampler` from `dimod`
- Outputs selected assets and solution energy

---

## 📦 Files Included

| File           | Description                            |
|----------------|----------------------------------------|
| `app.py`       | Streamlit app using simulated annealing|
| `requirements.txt` | Python dependencies               |
| `README.md`    | Project overview (this file)           |

---

## 🔧 Setup Instructions

### 1. Install Python packages
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

---

## 🧪 Powered By

- [Streamlit](https://streamlit.io/)
- [dimod](https://docs.ocean.dwavesys.com/projects/dimod/en/latest/)
- [NumPy](https://numpy.org/)

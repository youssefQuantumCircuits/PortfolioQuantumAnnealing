# 🌌 Quantum Portfolio Optimizer

This is a Streamlit app that uses a **D-Wave quantum annealer** to solve a basic **portfolio optimization problem** over 5 assets.

---

## 📈 What It Does

- Lets you adjust **risk aversion** via a slider
- Translates a portfolio optimization objective into a **QUBO (Quadratic Unconstrained Binary Optimization)** problem
- Sends the QUBO to a **D-Wave quantum annealer**
- Outputs the **optimal asset selection** based on return and risk
- Displays **quantum sample energy** (solution quality)

---

## 📦 Files Included

| File        | Description                             |
|-------------|-----------------------------------------|
| `app.py`    | Streamlit app using D-Wave Annealer     |
| `requirements.txt` | Python dependencies              |
| `README.md` | Project overview (this file)            |

---

## 🔧 Setup Instructions

### 1. Install Python packages
```bash
pip install -r requirements.txt
```

### 2. Set your D-Wave API token
You must create a free account on [D-Wave Leap](https://cloud.dwavesys.com/leap) and set your token:
```bash
export DWAVE_API_TOKEN="your-api-token"
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## 🧠 Background

This app demonstrates **quantum-enhanced optimization** using real quantum hardware. It applies a classical finance problem (portfolio optimization) to a modern quantum framework (QUBO solved by annealing).

---

## 👤 Author

Built by **Youssef Mahmoud**  
Specialist in quantum computing, AI, and finance

---

## 🧪 Powered By

- [D-Wave Ocean SDK](https://docs.ocean.dwavesys.com/)
- [Streamlit](https://streamlit.io/)
- [NumPy](https://numpy.org/)

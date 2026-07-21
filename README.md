# 📊 Markov Chain Risk Engine

A Streamlit-based quantitative credit risk application that estimates mortgage delinquency transition probabilities using **Fannie Mae Single-Family Servicing Data**. The application models borrower migration across delinquency states using a Markov Chain framework and estimates portfolio-level expected loss through absorbing state analysis.

---

## 🚀 Live Demo

```
https://markov-chain-risk-engine.streamlit.app/
```

---

## 📌 Overview

This project demonstrates how **Markov Chains** can be applied in credit risk modeling to estimate borrower migration between delinquency states over time.

The application:

- Estimates transition probability matrices using Maximum Likelihood Estimation (MLE)
- Allows dynamic calibration using a rolling lookback window
- Forecasts multi-period transition probabilities using the Chapman-Kolmogorov equation
- Models **90+ Days Delinquent** as an absorbing state
- Estimates portfolio Expected Loss using Loss Given Default (LGD)

The application is built entirely in **Python** using **Streamlit**, making it interactive and easy to deploy.

---

# Features

### 📂 Data Upload

Upload Fannie Mae Single-Family Servicing loan performance data in CSV format.

Supported dataset:

- Fannie Mae Single-Family Loan Performance Dataset

---

### 📈 Transition Matrix Estimation

Estimates one-step transition probabilities using Maximum Likelihood Estimation.

States:

- Current (C)
- 30 Days Delinquent (D30)
- 60 Days Delinquent (D60)
- 90+ Days Delinquent (D90)

---

### 📅 Adjustable Lookback Window

Estimate transition matrices using only the most recent:

- 3 Months
- 6 Months
- 12 Months

This enables comparison between short-term and long-term borrower behaviour.

---

### 🔄 Multi-Step Transition Forecasting

Uses the Chapman-Kolmogorov equation

\[
P^n
\]

to estimate transition probabilities over multiple months.

Example:

- Probability(Current → 90+ Days Delinquent in 24 months)
- Probability(D30 → Current in 12 months)

---

### 🚨 Absorbing State Analysis

Creates an absorbing Markov Chain where:

```
90+ Days Delinquent
```

becomes an absorbing state.

Useful for:

- Credit Risk
- Default Modeling
- Loss Forecasting

---

### 💰 Expected Loss Calculation

Portfolio Expected Loss is calculated using:

\[
Expected\ Loss = EAD \times PD \times LGD
\]

where

- Exposure at Default (Portfolio Balance)
- Probability of Default (Derived from Markov Chain)
- Loss Given Default (User Adjustable)

---

### 📊 Portfolio Dashboard

Displays:

- Total Portfolio Value
- Average Loan Balance
- Number of Loans
- Portfolio Delinquency Rate

---

# Mathematical Background

## Transition Matrix

Transition probabilities are estimated using

\[
\hat P_{ij}
=
\frac{N_{ij}}
{\sum_k N_{ik}}
\]

where

- \(N_{ij}\) = observed transitions from state \(i\) to state \(j\)

---

## Chapman-Kolmogorov Equation

Multi-step transition probabilities are calculated as

\[
P^n
\]

where

- \(P\) is the one-step transition matrix
- \(n\) is the forecast horizon

---

## Absorbing Markov Chain

The model also estimates an absorbing transition matrix by treating

```
90+ Days Delinquent
```

as the terminal default state.

---

# Technologies Used

- Python
- Streamlit
- Pandas
- NumPy

---

# Project Structure

```
Markov-Chain-Risk-Engine/

│
├── app.py
├── requirements.txt
├── README.md
│
├── assets/
│   └── github-photo.jpg
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/sub-shi/Markov-Chain-Risk-Engine.git
```

Move into the project

```bash
cd Markov-Chain-Risk-Engine
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run locally

```bash
streamlit run app.py
```

---

# Data Source

Fannie Mae Single-Family Loan Performance Data

https://datadynamics.fanniemae.com/data-dynamics/#/downloadLoanData/Single-Family

---

# Example Workflow

1. Upload servicing data
2. Select Lookback Period
3. Estimate Transition Matrix
4. Forecast Transition Probabilities
5. Analyze Absorbing State
6. Estimate Portfolio Expected Loss

---

# Future Improvements

- Continuous-time Markov Chains
- Macroeconomic scenario analysis
- Probability of Default term structure
- Monte Carlo portfolio simulation
- Lifetime Expected Credit Loss (IFRS 9 / CECL)
- Loan-level forecasting
- Portfolio segmentation by credit characteristics
- Transition matrix heatmaps
- Downloadable reports

---

# Author

**Subrat Sethi**

Developer & Quantitative Reseacher

GitHub:
https://github.com/sub-shi

LinkedIn:
https://linkedin.com/in/subrat-sethi-0a7934208

X (Twitter):
https://x.com/syn_engineer

---

# License

This project is intended for educational, research, and portfolio purposes.

Dataset © Fannie Mae.
import streamlit as st
import pandas as pd
import numpy as np


def clean_and_standardize_data(df):
    """
    Standardize uploaded Fannie Mae servicing data.
    """

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all").copy()

    parsed = pd.DataFrame()

    # -----------------------------
    # Loan ID
    # -----------------------------
    parsed["Loan_ID"] = df["1"].astype(str).str.strip()

    # -----------------------------
    # Reporting Month (MMYYYY)
    # -----------------------------
    parsed["Month"] = pd.to_datetime(
        df["2"].astype(str).str.strip(),
        format="%m%Y",
        errors="coerce"
    )

    # -----------------------------
    # Current UPB (Balance)
    # -----------------------------
    parsed["Balance"] = (
        pd.to_numeric(df["11"], errors="coerce")
        .fillna(0)
    )

    # -----------------------------
    # Raw Delinquency Status
    # -----------------------------
    parsed["Status"] = (
        df["39"]
        .astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
    )

    # -----------------------------
    # Map Fannie Mae delinquency codes
    # -----------------------------
    def map_status(x):

        try:
            x = int(float(x))
        except:
            x = str(x).upper()

            if x in ["RA", "XX", "X"]:
                return "D90"

            return "C"

        if x <= 0:
            return "C"
        elif x == 1:
            return "D30"
        elif x == 2:
            return "D60"
        else:
            return "D90"

    parsed["Status"] = parsed["Status"].apply(map_status)
    # -----------------------------
    # Remove bad records
    # -----------------------------
    parsed = parsed.dropna(subset=["Loan_ID", "Month"])

    parsed["Balance"] = parsed["Balance"].clip(lower=0)

    return parsed


# --- PAGE CONFIG ---
st.set_page_config(page_title="Markov Chain Single-Family Model", layout="wide")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.title("Model Settings")
    
    lookback_period = st.slider("Lookback Period (Months)", min_value=1, max_value=12, value=3, step=1)
    lgd_value = st.number_input("Loss Given Default (LGD)", min_value=0.0, max_value=1.0, value=0.40, step=0.05, format="%.2f")
    
    st.markdown("---")
    st.subheader("Data Source:")
    st.markdown("[Fannie Mae Single-Family Loan Data](https://datadynamics.fanniemae.com/data-dynamics/#/downloadLoanData/Single-Family)")
    
    st.subheader("Example of Data Format to Upload:")
    st.markdown("[Google Drive Sample Data Chunks](https://drive.google.com/drive/folders/1S3JNf7vvBQLW22e7MAd19z7wu0uInwR1)")
    
    st.markdown("---")
    st.title("About the Author")
    st.image("assets/github-photo.jpg", width=180)
    st.markdown("""
    **Subrat Sethi** <br>
    Developer & Quantitative Researcher
    * [GitHub Profile](https://github.com/sub-shi)
    * [LinkedIn Profile](https://linkedin.com/in/subrat-sethi-0a7934208)
    * [X Profile](https://x.com/syn_engineer)
    """, unsafe_allow_html=True)

# --- 1. HEADER SECTION ---
st.title("Markov Chain Single-Family Model")

st.write("""
This app enables the user to develop a markov chain model for single-family loans available from Fannie Mae. 
Linked on the sidebar is the original data source along with an example of chunks to upload to the application 
*(remember chunk uploads must be contiguous for state transitions!)*.
""")

st.write("""
This app is hosted using Streamlit, large data may not be processed on this server and is best handled on a local instance of this app. 
Please clone this repository to run the app locally on your machine:
""")
st.markdown("[https://github.com/sub-shi/Markov-Chain-Risk-Engine](https://github.com/sub-shi/Markov-Chain-Risk-Engine)")

st.markdown("---")

# --- 2. STATISTICAL PARAMETER ESTIMATION ---
st.header("Statistical Parameter Estimation")

with st.expander("Markov Chain Parameter Estimation", expanded=False):
    st.subheader("Transition Matrix Estimation Methodology")
    st.write("""
    Given a set of possible states $S = \{\\text{Current}, \\text{30 Days Delinquent}, \\text{60 Days Delinquent}, \\text{90+ Days Delinquent}\}$, 
    the goal is to estimate a transition probability matrix $P$, where each entry $P_{ij}$ represents 
    the probability of transitioning from state $i$ to state $j$.
    """)
    
    st.subheader("Maximum Likelihood Estimation (MLE)")
    st.write("""
    Using observed data, the transition probabilities can be estimated using Maximum Likelihood Estimation (MLE). 
    Suppose $N_{ij}$ denotes the number of observed transitions from state $i$ to state $j$ across all loan sequences. 
    The MLE for the transition probability $P_{ij}$ is given by:
    """)
    st.latex(r"\hat{P}_{ij} = \frac{N_{ij}}{\sum_k N_{ik}}")
    st.write("""
    where $\sum_k N_{ik}$ is the total number of transitions observed from state $i$ to any other state. 
    This ensures that each row in the transition matrix sums to 1.
    """)
    
    st.subheader("Step-by-Step Process")
    st.markdown("""
    1. **Categorize Loan States:** Each loan's delinquency status is categorized into discrete states such as *Current*, *30 Days Delinquent*, etc.
    2. **Calculate Transitions:** For each loan, calculate transitions between these states by observing consecutive periods.
    3. **Estimate Transition Matrix:** Using MLE, we compute the probability of moving from one state to another as shown above.
    """)

st.markdown("""
**More on Parameter Estimation:** [Maximum Likelihood Estimation Article](https://www.sciencedirect.com/topics/engineering/maximum-likelihood-estimation)
""")

# --- 3. LOOKBACK PERIOD EXPANDER ---
st.header("Lookback Period for Parameter Estimation")

with st.expander("Lookback Period Selection Explanation", expanded=False):
    st.write("""
    The lookback period determines the number of consecutive months used to estimate transition probabilities. Here's why different windows might be preferred:
    """)
    st.markdown("""
    * **3-6 Months:** Often chosen for shorter-term risk assessment, capturing recent borrower behavior and adapting to economic shifts.
    * **12 Months:** Common in annual risk assessments, this window smooths seasonal patterns and is often used in regulatory models.
    * **18+ Months:** Longer periods may provide stable trends in risk but can dilute recent changes, so they're usually used in strategic, longer-term models.
    """)

st.markdown("---")

# --- 4. DATA UPLOAD SECTION ---
st.header("Single-Family Data Upload")

uploaded_file = st.file_uploader(
    "Upload your loan data CSV file (in chunks)",
    type=["csv", "txt"]
)

@st.cache_data
def load_sample_data():
    """Generates a small default dataset."""
    return pd.DataFrame({
        "Loan_ID": [101,101,101,102,102,102,103,103,103,104,104,104],
        "Month": [1,2,3,1,2,3,1,2,3,1,2,3],
        "Status": ["C","D30","D60","C","C","D30","D90","D90","D90","C","C","C"],
        "Balance": [250000]*3 + [300000]*3 + [150000]*3 + [200000]*3
    })


if uploaded_file is not None:

    uploaded_file.seek(0)

    # Read the uploaded pipe-delimited file
    df = pd.read_csv(
        uploaded_file,
        sep="|",
        header=0,
        low_memory=False
    )

    # Remove exported pandas index column if present
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # If only one column was read, retry as a comma-separated CSV
    if df.shape[1] == 1:
        uploaded_file.seek(0)

        df = pd.read_csv(
            uploaded_file,
            sep=",",
            low_memory=False
        )

    # Clean and standardize
    df = clean_and_standardize_data(df)

else:

    st.info(
        " No file uploaded yet. Using built-in sample portfolio data for demonstration."
    )

    df = load_sample_data()

# --- 5. CORE COMPUTATION ENGINE ---
def compute_transition_matrix(dataframe):

    dataframe = dataframe.copy()

    # Ensure Month is datetime
    dataframe["Month"] = pd.to_datetime(
        dataframe["Month"],
        errors="coerce"
    )

    dataframe = dataframe.dropna(subset=["Month"])

    # --------------------------------------------------
    # Apply lookback period only for real date data
    # --------------------------------------------------
    if dataframe["Month"].max().year > 2000:

        latest_month = dataframe["Month"].max()

        cutoff = latest_month - pd.DateOffset(
            months=lookback_period - 1
        )

        dataframe = dataframe[
            dataframe["Month"] >= cutoff
        ]

    # Sort chronologically
    dataframe = (
        dataframe
        .sort_values(["Loan_ID", "Month"])
        .reset_index(drop=True)
    )

    # Create next month's state
    dataframe["Next_Status"] = (
        dataframe
        .groupby("Loan_ID")["Status"]
        .shift(-1)
    )

    # Remove last observation of each loan
    pairs = dataframe.dropna(subset=["Next_Status"])

    states = ["C", "D30", "D60", "D90"]

    # Transition counts
    ct = pd.crosstab(
        pairs["Status"],
        pairs["Next_Status"]
    ).reindex(
        index=states,
        columns=states,
        fill_value=0
    )

    # Convert counts to probabilities
    matrix_prob = ct.div(
        ct.sum(axis=1).replace(0, 1),
        axis=0
    )

    # Handle empty rows
    for state in states:
        if ct.loc[state].sum() == 0:
            matrix_prob.loc[state] = 0
            matrix_prob.loc[state, state] = 1

    return matrix_prob


# Compute transition matrix
# Original transition matrix
transition_matrix = compute_transition_matrix(df)

# Absorbing version
p_matrix = transition_matrix.copy()

p_matrix.loc["D90"] = [0, 0, 0, 1]


# --- 6. PORTFOLIO SUMMARY STATISTICS ---
st.header("Portfolio Summary Statistics")

col1, col2, col3, col4 = st.columns(4)

# Latest record for each loan
latest_df = (
    df
    .sort_values(["Loan_ID", "Month"])
    .groupby("Loan_ID")
    .tail(1)
    .reset_index(drop=True)
)

# Portfolio metrics
total_loans = len(latest_df)

total_val = latest_df["Balance"].sum()

avg_bal = latest_df["Balance"].mean()

delinquency_rate = (
    latest_df["Status"]
    .isin(["D30", "D60", "D90"])
    .mean()
)

# Display metrics
col1.metric("Unique Loans", f"{total_loans:,}")
col2.metric("Total Loan Value", f"${total_val:,.2f}")
col3.metric("Average Loan Balance", f"${avg_bal:,.2f}")
col4.metric("Delinquency Rate", f"{delinquency_rate:.2%}")

# --- 7. FORWARD LOOKING PROBABILITY & EXPECTATION ---
st.header("Forward Looking Probability & Expectation")

with st.expander("Chapman-Kolmogorov Equation Explanation", expanded=False):
    st.write("""
    The Chapman-Kolmogorov equation allows us to compute the probability of transitioning from one state to another over multiple steps. Given a transition matrix $P$ and a number of steps $n$, the probability of transitioning from state $i$ to state $j$ in $n$ steps is given by $P_{ij}^n$.
    """)
    st.write("""
    This can be computed by raising the transition matrix $P$ to the power $n$.
    """)

st.write("""
Noticing after an arbitrary number of transitions the transition probability is constant? This is expected and is an example of how we can estimate the stationary distribution! Further, you may also notice that this isn't the case for the transition matrix consisting of an absorbing state as it violates the positive recurrence assumption. For more information and a mathematical definition see the expander:
""")

with st.expander("Understanding Steady-State Probabilities and Long-Run Equilibrium", expanded=False):
    st.subheader("Steady-State Probabilities and Long-Run Equilibrium")
    st.write("""
    In a Markov chain, **steady-state probabilities** represent the long-run probability of being in each state, regardless of the initial state. This concept is also known as the **long-run equilibrium**.
    """)
    st.subheader("Mathematical Definition")
    st.latex(r"\pi = \pi P")
    st.latex(r"\pi = \lim_{n \to \infty} \pi_0 P^n")

# --- 8. TRANSITION CALCULATOR ---
st.subheader("Transition Probability Calculator")

c1, c2, c3 = st.columns(3)
with c1:
    start_state = st.selectbox("Select Start State", transition_matrix.index, index=0)
with c2:
    end_state = st.selectbox("Select End State", transition_matrix.columns, index=3)
with c3:
    months = st.number_input("Number of Months", min_value=1, max_value=120, value=1)

p_n = np.linalg.matrix_power(transition_matrix.values, int(months))
forecast_df = pd.DataFrame(
    p_n,
    index=transition_matrix.index,
    columns=transition_matrix.columns
)

target_prob = forecast_df.loc[start_state, end_state]
st.info(f"Probability from **{start_state}** to **{end_state}** in {months} month(s): **{target_prob:.2%}**")

# --- 9. ABSORBING STATE SECTION ---
st.header("90+ Days Delinquent as Absorbing State")

with st.expander("Absorbing State Modeling for 90+ Days Delinquent", expanded=False):
    st.write("**Absorbing Transition Matrix:**")

    transition_display = p_matrix.rename(
        index={
            "C": "Current",
            "D30": "30 Days Delinquent",
            "D60": "60 Days Delinquent",
            "D90": "90+ Days Delinquent"
        },
        columns={
            "C": "Current",
            "D30": "30 Days Delinquent",
            "D60": "60 Days Delinquent",
            "D90": "90+ Days Delinquent"
        }
    )

    st.dataframe(
        transition_display.style.format("{:.4f}"),
        use_container_width=True
    )

    st.write("**Absorbing State Row (90+ Days Delinquent):**")

    absorbing_row = transition_display.loc[["90+ Days Delinquent"]]

    st.dataframe(
        absorbing_row.style.format("{:.4f}"),
        use_container_width=True
    )

st.header("Absorbing State Transition Probability Calculator")

c1, c2, c3 = st.columns(3)
with c1:
    abs_start_state = st.selectbox("Select Start State (Absorbing Matrix)", p_matrix.index, index=0, key="abs_start")
with c2:
    abs_end_state = st.selectbox("Select End State (Absorbing Matrix)", p_matrix.columns, index=3, key="abs_end")
with c3:
    abs_months = st.number_input("Number of Months (Absorbing Horizon)", min_value=1, max_value=120, value=1, key="abs_months")

abs_p_n = np.linalg.matrix_power(p_matrix.values, int(abs_months))
abs_forecast_df = pd.DataFrame(abs_p_n, index=p_matrix.index, columns=p_matrix.columns)

abs_target_prob = abs_forecast_df.loc[abs_start_state, abs_end_state]
st.info(f"Absorbing Transition Probability from **{abs_start_state}** to **{abs_end_state}** in {abs_months} month(s): **{abs_target_prob:.2%}**")


# --- 10. EXPECTED LOSS CALCULATION ---
# Latest portfolio state
latest_df = (
    df.sort_values(["Loan_ID", "Month"])
      .groupby("Loan_ID")
      .tail(1)
)

# Portfolio distribution by state
portfolio = (
    latest_df["Status"]
        .value_counts(normalize=True)
        .reindex(["C", "D30", "D60", "D90"], fill_value=0)
)

# Portfolio-weighted probability of being in D90
default_probability = (
    portfolio["C"]   * abs_forecast_df.loc["C", "D90"] +
    portfolio["D30"] * abs_forecast_df.loc["D30", "D90"] +
    portfolio["D60"] * abs_forecast_df.loc["D60", "D90"] +
    portfolio["D90"] * abs_forecast_df.loc["D90", "D90"]
)

expected_loss = total_val * default_probability * lgd_value

st.subheader(
    f"Expected Loss based on LGD ({lgd_value:.0%}): ${expected_loss:,.2f}"
)
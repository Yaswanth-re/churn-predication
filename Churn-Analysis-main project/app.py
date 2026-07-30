"""Interactive customer-churn risk explorer for Streamlit Community Cloud."""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Churn Signal", page_icon="◒", layout="wide")

DATA_FILE = Path(__file__).with_name("Prediction_Data.xlsx")
TARGET = "Customer_Status"
EXCLUDED_COLUMNS = ["Customer_ID", "Churn_Category", "Churn_Reason", TARGET]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Manrope:wght@400;500;600;700&display=swap');
    :root { --ink:#11221c; --pine:#164a3b; --leaf:#a6c95d; --mist:#edf2e8; --chalk:#fbfcf7; --coral:#e87252; }
    .stApp { background: var(--chalk); color: var(--ink); font-family: Manrope, sans-serif; }
    .block-container { max-width: 1280px; padding-top: 2.2rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-family: Fraunces, Georgia, serif !important; color: var(--ink) !important; letter-spacing:-.035em; }
    h1 { font-size:clamp(2.9rem, 7vw, 5.5rem) !important; line-height:.91 !important; margin-bottom:.55rem !important; }
    [data-testid="stSidebar"] { background:#143e33; }
    [data-testid="stSidebar"] * { color:#f7f9ef !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color:#11221c !important; }
    .eyebrow { font: 500 .72rem 'DM Mono', monospace; letter-spacing:.11em; text-transform:uppercase; color:#367363; }
    .hero { border-bottom:1px solid #cbd6c9; padding-bottom:1.6rem; margin-bottom:1.4rem; }
    .hero p { max-width:660px; font-size:1.04rem; line-height:1.6; margin:0; color:#416056; }
    .metric-card { background:var(--mist); border-left:4px solid var(--leaf); padding:1rem 1.15rem; min-height:92px; }
    .metric-label { color:#477164; font:500 .67rem 'DM Mono', monospace; letter-spacing:.08em; text-transform:uppercase; }
    .metric-value { color:var(--ink); font:700 2rem Fraunces,serif; line-height:1.15; margin-top:.22rem; }
    .risk-note { background:#fff4ed; border:1px solid #f5c9b9; padding:1rem 1.2rem; color:#75402f; }
    .stButton > button { border-radius:0; border:0; background:var(--pine); color:white; font-weight:700; padding:.55rem 1.1rem; }
    .stButton > button:hover { background:#0f3027; color:white; }
    [data-testid="stDataFrame"] { border:1px solid #d6e0d4; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data():
    churn = pd.read_excel(DATA_FILE, sheet_name="vw_churndata")
    joiners = pd.read_excel(DATA_FILE, sheet_name="vw_joindata")
    return churn, joiners


@st.cache_resource(show_spinner="Training the risk model…")
def build_model(churn: pd.DataFrame):
    model_data = churn.dropna(subset=[TARGET]).copy()
    features = model_data.drop(columns=EXCLUDED_COLUMNS)
    target = model_data[TARGET].eq("Churned").astype(int)
    categorical = features.select_dtypes(include="object").columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]
    transformer = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical), ("numeric", "passthrough", numeric)],
        remainder="drop",
    )
    pipeline = Pipeline(
        [("preprocess", transformer), ("model", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced", n_jobs=-1))]
    )
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=.20, random_state=42, stratify=target)
    pipeline.fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, pipeline.predict(X_test))
    matrix = confusion_matrix(y_test, pipeline.predict(X_test))
    return pipeline, features.columns.tolist(), test_accuracy, matrix


def calculate_joiner_risk(model, columns, joiners):
    records = joiners.drop(columns=["Customer_ID", "Customer_Status", "Churn_Category", "Churn_Reason"], errors="ignore").copy()
    records = records.reindex(columns=columns)
    scored = joiners.copy()
    scored["Churn_risk"] = model.predict_proba(records)[:, 1]
    scored["Risk_band"] = pd.cut(scored["Churn_risk"], bins=[-0.01, .30, .60, 1.0], labels=["Low", "Watch", "High"])
    return scored


churn, joiners = load_data()
model, feature_columns, test_accuracy, matrix = build_model(churn)
scored_joiners = calculate_joiner_risk(model, feature_columns, joiners)

with st.sidebar:
    st.markdown("## Churn Signal")
    st.caption("Customer-retention workbench")
    page = st.radio("Navigate", ["Portfolio pulse", "Joiner risk queue", "Model check"], label_visibility="collapsed")
    st.divider()
    st.caption("Data source")
    st.write("Prediction_Data.xlsx")
    st.caption("Model")
    st.write("Random Forest · 250 trees")

if page == "Portfolio pulse":
    churn_rate = churn[TARGET].eq("Churned").mean()
    avg_charge = churn.loc[churn[TARGET].eq("Churned"), "Monthly_Charge"].mean()
    short_tenure_rate = churn.loc[churn["Tenure_in_Months"] <= 12, TARGET].eq("Churned").mean()
    st.markdown('<section class="hero"><div class="eyebrow">Retention overview · historic customer base</div><h1>See the customers<br>starting to slip.</h1><p>A focused view of churn signals in the supplied customer data, designed to direct retention attention before a customer leaves.</p></section>', unsafe_allow_html=True)
    columns = st.columns(4)
    values = [("Customers analysed", f"{len(churn):,}"), ("Observed churn", f"{churn_rate:.1%}"), ("Churned monthly charge", f"${avg_charge:,.0f}"), ("≤12-month churn", f"{short_tenure_rate:.1%}")]
    for column, (label, value) in zip(columns, values):
        column.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        by_contract = (churn.assign(Churned=churn[TARGET].eq("Churned")).groupby("Contract", as_index=False)["Churned"].mean().sort_values("Churned", ascending=False))
        chart = px.bar(by_contract, x="Churned", y="Contract", orientation="h", text=by_contract["Churned"].map("{:.1%}".format), color_discrete_sequence=["#164a3b"])
        chart.update_layout(title="Contract type is the clearest retention signal", xaxis_title="Churn rate", yaxis_title="", height=350, margin=dict(l=0,r=0,t=55,b=0), paper_bgcolor="#fbfcf7", plot_bgcolor="#fbfcf7")
        chart.update_xaxes(tickformat=".0%", showgrid=True, gridcolor="#dfe8dd")
        chart.update_yaxes(autorange="reversed")
        st.plotly_chart(chart, use_container_width=True)
    with right:
        state = (churn.assign(Churned=churn[TARGET].eq("Churned")).groupby("State", as_index=False)["Churned"].mean().nlargest(8, "Churned"))
        chart = px.bar(state, x="State", y="Churned", color="Churned", color_continuous_scale=["#a6c95d", "#e87252"])
        chart.update_layout(title="States with the highest observed churn", xaxis_title="", yaxis_title="Churn rate", height=350, margin=dict(l=0,r=0,t=55,b=0), paper_bgcolor="#fbfcf7", plot_bgcolor="#fbfcf7", coloraxis_showscale=False)
        chart.update_yaxes(tickformat=".0%", gridcolor="#dfe8dd")
        st.plotly_chart(chart, use_container_width=True)
    st.subheader("What the data points to")
    st.info("Month-to-month plans and short-tenure customers have higher observed churn in this dataset. Start outreach there, then use the joiner risk queue to prioritize new customers needing early support.")

elif page == "Joiner risk queue":
    st.markdown('<section class="hero"><div class="eyebrow">Prediction queue · 411 incoming customers</div><h1>Turn early signals<br>into timely outreach.</h1><p>Score new customers with the model trained on historical churn outcomes. Use this queue to guide human review, not to make automated decisions.</p></section>', unsafe_allow_html=True)
    high_risk = (scored_joiners["Risk_band"] == "High").sum()
    c1, c2, c3 = st.columns(3)
    for col, label, value in [(c1,"Joiners scored", f"{len(scored_joiners):,}"), (c2,"High-risk joiners", f"{high_risk:,}"), (c3,"Highest predicted risk", f"{scored_joiners['Churn_risk'].max():.0%}")]:
        col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    risk_filter = st.multiselect("Show risk bands", ["High", "Watch", "Low"], default=["High", "Watch"])
    display = scored_joiners.loc[scored_joiners["Risk_band"].astype(str).isin(risk_filter), ["Customer_ID", "State", "Age", "Tenure_in_Months", "Contract", "Internet_Type", "Payment_Method", "Monthly_Charge", "Churn_risk", "Risk_band"]].sort_values("Churn_risk", ascending=False)
    display["Churn_risk"] = display["Churn_risk"].map("{:.1%}".format)
    st.dataframe(display, use_container_width=True, hide_index=True, column_config={"Churn_risk": st.column_config.TextColumn("Predicted churn risk")})
    st.markdown('<div class="risk-note"><strong>Responsible use:</strong> Model probabilities are estimates based on the provided training data. Review recommendations with people, monitor performance after launch, and avoid using sensitive attributes to deny service or impose adverse outcomes.</div>', unsafe_allow_html=True)
    st.download_button("Download filtered queue as CSV", display.to_csv(index=False).encode("utf-8"), file_name="churn_risk_queue.csv", mime="text/csv")

else:
    st.markdown('<section class="hero"><div class="eyebrow">Model evaluation · held-out historic data</div><h1>Check the signal<br>before acting on it.</h1><p>The model is re-trained from the included historical data on each fresh app build. This panel reports its held-out test accuracy, not a production guarantee.</p></section>', unsafe_allow_html=True)
    c1, c2 = st.columns([.55,.45])
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Held-out accuracy</div><div class="metric-value">{test_accuracy:.1%}</div></div>', unsafe_allow_html=True)
        st.caption("Accuracy is useful context but can hide uneven performance between churned and retained customers. Review precision, recall, and outcomes periodically before operational use.")
    with c2:
        matrix_df = pd.DataFrame(matrix, index=["Actual stayed", "Actual churned"], columns=["Predicted stayed", "Predicted churned"])
        st.markdown("#### Confusion matrix")
        st.dataframe(matrix_df, use_container_width=True)
    st.markdown("#### Feature coverage")
    st.write(f"The model uses {len(feature_columns)} supplied behavioral, account, service, and billing fields. Customer IDs, churn reasons, and churn categories are excluded from model inputs.")

import os

# Limit math-library threads BEFORE numpy/pandas/sklearn load.
# This avoids "OpenBLAS error: Memory allocation still failed" on computers with limited RAM.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import base64                      # FIX 1: was missing (needed for the PDF preview)
import io
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

st.set_page_config(page_title="Job Acceptance Prediction", layout="wide")

APP_DIR = Path(r"C:\Users\User\OneDrive\Desktop\Project_3\HR_Job_Placement_cleaned.csv").resolve().parent
# FIX 2: the file on your disk is spelled "..._cleaned.csv" (small c); both spellings are listed
DATA_NAMES = ["HR_Job_Placement_cleaned.csv", "HR_Job_Placement_Cleaned.csv"]
FEATURES = [
    "age_years", "gender", "ssc_percentage", "hsc_percentage", "degree_percentage",
    "degree_specialization", "technical_score", "aptitude_score", "communication_score",
    "skills_match_percentage", "certifications_count", "internship_experience",
    "years_of_experience", "career_switch_willingness", "relevant_experience",
    "previous_ctc_lpa", "expected_ctc_lpa", "company_tier", "job_role_match",
    "competition_level", "bond_requirement", "notice_period_days", "layoff_history",
    "employment_gap_months", "relocation_willingness",
]


def find_data_file():
    """Looks for the dataset next to app.py first, then in the folder where Streamlit was started."""
    for folder in (APP_DIR, Path.cwd()):
        for name in DATA_NAMES:
            if (folder / name).exists():
                return folder / name
    return None


def pdf_page_count(pdf_bytes):
    """FIX 3: this function was missing. Returns the page count, or None if 'pypdf' is not installed."""
    try:
        from pypdf import PdfReader
        return len(PdfReader(io.BytesIO(pdf_bytes)).pages)
    except Exception:
        return None


# ------------------------------------------------------------------ data and models
@st.cache_data
def load_data(source):
    """source = a file path, or the raw bytes of an uploaded CSV."""
    df = pd.read_csv(io.BytesIO(source) if isinstance(source, bytes) else source)
    for c in df.select_dtypes(exclude="number").columns:
        df[c] = df[c].str.strip().str.title()
    df = df.drop_duplicates().reset_index(drop=True)
    df["placed"] = (df["status"] == "Placed").astype(int)
    return df


def gb_pipeline(X):
    text_cols = X.select_dtypes(exclude="number").columns.tolist()
    return make_pipeline(
        make_column_transformer((OneHotEncoder(handle_unknown="ignore"), text_cols), remainder="passthrough"),
        HistGradientBoostingClassifier(class_weight="balanced", random_state=42))


@st.cache_resource
def train_best_model(source):
    """Gradient Boosting trained on all candidates (used on the prediction page)."""
    df = load_data(source)
    model = gb_pipeline(df[FEATURES])
    model.fit(df[FEATURES], df["placed"])
    return model


@st.cache_data
def compute_kpis(source):
    """KPI numbers for the KPI page.
    Each candidate gets an out-of-fold Gradient Boosting score, so no candidate is scored by a model that saw them."""
    df = load_data(source)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    prob = cross_val_predict(gb_pipeline(df[FEATURES]), df[FEATURES], df["placed"],
                             cv=cv, method="predict_proba")[:, 1]
    likely_offer = prob >= 0.5                      # candidates who look like they will get / accept an offer
    dropped = likely_offer & (df["placed"].values == 0)
    interview = df[["technical_score", "aptitude_score", "communication_score"]].mean(axis=1)
    return {
        "total": len(df),
        "placement_rate": df["placed"].mean() * 100,
        "acceptance_rate": df["placed"].mean() * 100,
        "avg_interview": interview.mean(),
        "avg_skills": df["skills_match_percentage"].mean(),
        "dropout_rate": dropped.sum() / likely_offer.sum() * 100,
        "high_risk": (prob < 0.2).mean() * 100,
    }


@st.cache_data
def compare_models(source):
    """Trains 5 models on an 80/20 split and returns the metrics table and the best model's confusion matrix."""
    df = load_data(source)
    X = df[FEATURES].copy()
    for c in X.columns:
        X[c] = X[c].fillna(X[c].median()) if pd.api.types.is_numeric_dtype(X[c]) else X[c].fillna("Unknown")
    X = pd.get_dummies(X, drop_first=True)
    y = df["placed"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "Gradient Boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=5, class_weight="balanced",
                                                n_jobs=1, random_state=42),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
    }
    rows, cms = [], {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        prob = m.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        cms[name] = confusion_matrix(y_test, pred)
        rows.append({"Model": name, "Accuracy": accuracy_score(y_test, pred),
                     "Precision": precision_score(y_test, pred), "Recall": recall_score(y_test, pred),
                     "F1": f1_score(y_test, pred), "ROC-AUC": roc_auc_score(y_test, prob)})
    res = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).round(4).reset_index(drop=True)
    return res, cms[res.iloc[0]["Model"]]


# ------------------------------------------------------------------ sidebar navigation
st.sidebar.title("Menu")
page = st.sidebar.radio("Go to page", ["1. Job Prediction", "2. ML Analytics", "3. Key Performance Indicators"])


# ------------------------------------------------------------------ where does the data come from?
def get_data_source():
    """Order: uploaded file -> typed path -> file next to app.py / in the current folder."""
    with st.sidebar.expander("Dataset", expanded=find_data_file() is None):
        uploaded = st.file_uploader("Upload the CSV", type="csv")
        typed = st.text_input("...or type the full path to the CSV")
    if uploaded is not None:
        return uploaded.getvalue()
    if typed.strip():
        p = Path(typed.strip().strip('"'))
        if p.exists():
            return p
        st.sidebar.error("That path does not exist.")
    return find_data_file()


SOURCE = get_data_source()
if SOURCE is None:
    st.title("Dataset not found")
    st.error("The app could not find HR_Job_Placement_cleaned.csv.")
    st.write("Pick **one** of these ways to give the app the data:")
    st.markdown(
        "1. **Upload** the CSV with the *Dataset* box in the left sidebar.\n"
        "2. **Type the full path** to the CSV in the same box, for example `C:\\Users\\User\\OneDrive\\Desktop\\Project_3\\HR_Job_Placement_cleaned.csv`.\n"
        f"3. **Copy the CSV** into this folder: `{APP_DIR}`"
    )
    st.stop()


# ================================================================== PAGE 1
if page == "1. Job Prediction":
    st.title("1. Job Prediction")
    st.write("Enter the candidate details and click **Predict** to see if the candidate will accept the job.")

    col1, col2 = st.columns(2)
    technical_score = col1.number_input("Technical score (0-100)", 0.0, 100.0, 70.0, step=1.0)
    skills_match = col1.number_input("Skills match % (0-100)", 0.0, 100.0, 75.0, step=1.0)
    years_exp = col1.number_input("Years of experience", 0, 40, 1, step=1)
    expected_ctc = col1.number_input("Expected salary (LPA)", 0.0, 100.0, 8.0, step=0.5)
    internship = col2.selectbox("Internship experience", ["Yes", "No"])
    job_role = col2.selectbox("Job role match", ["Matched", "Not Matched"])
    competition = col2.selectbox("Competition level", ["Low", "Medium", "High"])

    if st.button("Predict"):
        df = load_data(SOURCE)
        X = df[FEATURES]
        text_cols = X.select_dtypes(exclude="number").columns.tolist()

        # typical values for the details that are not asked
        candidate = X.median(numeric_only=True).to_dict()
        candidate.update(X[text_cols].mode().iloc[0].to_dict())
        candidate.update({
            "technical_score": technical_score,
            "skills_match_percentage": skills_match,
            "years_of_experience": years_exp,
            "expected_ctc_lpa": expected_ctc,
            "internship_experience": internship,
            "job_role_match": job_role,
            "competition_level": competition,
        })

        model = train_best_model(SOURCE)
        prob = model.predict_proba(pd.DataFrame([candidate])[FEATURES])[0, 1]

        if prob >= 0.5:
            st.success("The candidate WILL ACCEPT the job")
        else:
            st.error("The candidate will NOT accept the job")
        st.metric("Model score (Gradient Boosting)", f"{prob * 100:.1f}%")
        if technical_score <= 60:
            st.warning("Technical score is 60 or below. In this dataset no such candidate was placed.")


# ================================================================== PAGE 2
elif page == "2. ML Analytics":
    st.title("2. ML Analytics")
    df = load_data(SOURCE)

    # FIX 4: everything below is now indented INSIDE this elif block
    tab_analytics, tab_pdf = st.tabs(["Analytics", "Attach PDF"])

    # ---------------- TAB 1 ----------------
    with tab_analytics:
        sections = ["Candidate Performance Analysis",
                    "Placement & Acceptance Analysis",
                    "Interview & Evaluation Analysis"]
        b1, b2, b3 = st.columns(3)
        if b1.button("1. " + sections[0], use_container_width=True):
            st.session_state["ml_section"] = sections[0]
        if b2.button("2. " + sections[1], use_container_width=True):
            st.session_state["ml_section"] = sections[1]
        if b3.button("3. " + sections[2], use_container_width=True):
            st.session_state["ml_section"] = sections[2]

        selected = st.session_state.get("ml_section")
        if selected:
            st.header(selected)
            st.empty()          # intentionally empty for now
        st.divider()

    # ---------------- TAB 2 (FIX 5: now a separate tab, not nested inside tab 1) ----------------
    with tab_pdf:
        st.subheader("Attach a PDF file")
        uploader_key = f"pdf_uploader_{st.session_state.get('pdf_key', 0)}"
        uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf", key=uploader_key)
        if uploaded_pdf is not None:
            st.session_state["attached_pdf"] = {"name": uploaded_pdf.name, "bytes": uploaded_pdf.getvalue()}

        attached = st.session_state.get("attached_pdf")
        if attached is None:
            st.info("No PDF attached yet. Use the box above to attach one.")
        else:
            size_kb = len(attached["bytes"]) / 1024
            st.success(f"Attached: {attached['name']} ({size_kb:,.1f} KB)")
            pages = pdf_page_count(attached["bytes"])
            if pages:
                st.write(f"Pages: {pages}")

            st.download_button("Download PDF", data=attached["bytes"],
                               file_name=attached["name"], mime="application/pdf")

            with st.expander("Preview", expanded=True):
                if size_kb > 5000:
                    st.warning("The file is larger than 5 MB, so the preview is skipped. Use the download button.")
                else:
                    b64 = base64.b64encode(attached["bytes"]).decode()
                    st.markdown(
                        f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="650" '
                        f'type="application/pdf"></iframe>', unsafe_allow_html=True)

            if st.button("Remove PDF"):
                del st.session_state["attached_pdf"]
                st.session_state["pdf_key"] = st.session_state.get("pdf_key", 0) + 1   # resets the upload box
                st.rerun()


# ================================================================== PAGE 3
else:
    # FIX 6: the line below was not indented, which is a syntax error after "else:"
    st.title("3. Key Performance Indicators")
    df = load_data(SOURCE)

    with st.spinner("Calculating KPIs..."):
        k = compute_kpis(SOURCE)
    row1 = st.columns(4)
    row1[0].metric("Total Candidates", f"{k['total']:,}")
    row1[1].metric("Placement Rate (%)", f"{k['placement_rate']:.1f}%")
    row1[2].metric("Job Acceptance Rate (%)", f"{k['acceptance_rate']:.1f}%")
    row1[3].metric("Average Interview Score", f"{k['avg_interview']:.1f}")
    row2 = st.columns(3)
    row2[0].metric("Average Skills Match %", f"{k['avg_skills']:.1f}%")
    row2[1].metric("Offer Dropout Rate", f"{k['dropout_rate']:.1f}%")
    row2[2].metric("High-Risk Candidate %", f"{k['high_risk']:.1f}%")

    st.divider()
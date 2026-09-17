# Steam Game Success Predictor - 4 page Streamlit dashboard
# Replace your existing app/app.py with this file.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Steam Game Success Predictor", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "steam_logistic_regression.joblib"
PREPROCESSOR_PATH = BASE_DIR / "models" / "steam_preprocessor.joblib"
DATA_PATH = BASE_DIR / "data" / "processed" / "steam_game_success_final.csv"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "data" / "processed" / "feature_importance.csv"

@st.cache_resource
def load_model(): return joblib.load(MODEL_PATH)
@st.cache_resource
def load_preprocessor(): return joblib.load(PREPROCESSOR_PATH)
@st.cache_data
def load_data(): return pd.read_csv(DATA_PATH)
@st.cache_data
def load_feature_importance():
    return pd.read_csv(FEATURE_IMPORTANCE_PATH) if FEATURE_IMPORTANCE_PATH.exists() else pd.DataFrame()

try:
    model = load_model()
    preprocessor = load_preprocessor()
    data = load_data()
    feature_importance = load_feature_importance()
except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()

st.markdown("""
<style>
.stApp { background-color:#161925; }
[data-testid="stSidebar"] { background-color:#23395B; }
.main-title { text-align:center; font-size:42px; font-weight:700; margin-bottom:4px; }
.subtitle { text-align:center; color:#9ca3af; font-size:17px; margin-bottom:30px; }
.section-title { font-size:25px; font-weight:650; margin-top:10px; margin-bottom:18px; }
.success-box { padding:24px; border-radius:14px; background:#12351f; border:1px solid #238636; text-align:center; }
.failure-box { padding:24px; border-radius:14px; background:#3a1717; border:1px solid #da3633; text-align:center; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Steam Game Success Predictor")
page = st.sidebar.radio("Navigate", ["Prediction", "Model Analysis", "Model Comparison", "Visualizations"])
st.sidebar.markdown("---")
st.sidebar.caption("Machine Learning Internship Project")


def get_release_era(year):
    if 2006 <= year <= 2010: return "2006-2010"
    if 2011 <= year <= 2013: return "2011-2013"
    if 2014 <= year <= 2016: return "2014-2016"
    if 2017 <= year <= 2019: return "2017-2019"
    return "Unknown"


def clean_feature_label(label):
    label = str(label)
    for prefix in ["num__", "binary__", "categorical__", "developer_", "publisher_", "release_era_"]:
        label = label.replace(prefix, "")
    return label.replace("genre_", "Genre: ").replace("tag_", "Tag: ").replace("_", " ")


def get_contributions(input_df):
    transformed = preprocessor.transform(input_df)
    if hasattr(transformed, "toarray"): transformed = transformed.toarray()
    names = preprocessor.get_feature_names_out()
    contributions = transformed[0] * model.coef_[0]
    result = pd.DataFrame({"Feature": names, "Contribution": contributions})
    result["Absolute"] = result["Contribution"].abs()
    return result.sort_values("Absolute", ascending=False).head(8)

# PAGE 1
if page == "Prediction":
    st.markdown('<div class="main-title">Steam Game Success Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predict the likelihood of a Steam game becoming successful based on its characteristics.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Game Information</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        release_year = st.number_input("Release Year", 2006, 2019, 2018, 1)
        original_price = st.number_input("Original Price", 0.0, 200.0, 19.99, 0.01)
    with c2:
        achievements = st.number_input("Achievements", 0, 5000, 50, 1)
        achievements_missing = st.checkbox("Achievements information missing")
    with c3:
        developer = st.text_input("Developer", "Unknown")
        publisher = st.text_input("Publisher", "Unknown")

    release_era = get_release_era(release_year)
    st.caption(f"Release era: {release_era}")

    genres = ["action","adventure","casual","indie","rpg","simulation","strategy","sports","racing","free_to_play","massively_multiplayer","early_access","utilities"]
    tags = ["action","indie","adventure","rpg","strategy","casual","simulation","sports","racing","multiplayer","fps","shooter","singleplayer"]

    st.markdown('<div class="section-title">Genres</div>', unsafe_allow_html=True)
    selected_genres = st.multiselect("Select genres", genres)
    st.markdown('<div class="section-title">Tags</div>', unsafe_allow_html=True)
    selected_tags = st.multiselect("Select tags", tags)

    if st.button("ANALYZE GAME SUCCESS", use_container_width=True):
        row = {
            "release_year": release_year, "release_era": release_era,
            "achievements": achievements, "achievements_missing": int(achievements_missing),
            "original_price": original_price, "is_free": int(original_price == 0),
            "developer": developer.strip() or "Unknown", "publisher": publisher.strip() or "Unknown"
        }
        row.update({f"genre_{g}": int(g in selected_genres) for g in genres})
        row.update({f"tag_{t}": int(t in selected_tags) for t in tags})
        input_df = pd.DataFrame([row])
        try:
            transformed = preprocessor.transform(input_df)
            prediction = model.predict(transformed)[0]
            probabilities = model.predict_proba(transformed)[0]
            success = probabilities[1]*100
            not_success = probabilities[0]*100
            if prediction == 1:
                st.markdown(f'<div class="success-box"><h1>SUCCESSFUL</h1><h3>Probability of Success: {success:.1f}%</h3></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="failure-box"><h1>NOT SUCCESSFUL</h1><h3>Probability of Success: {success:.1f}%</h3></div>', unsafe_allow_html=True)
            a,b = st.columns(2)
            a.metric("Success Probability", f"{success:.1f}%")
            b.metric("Not Successful Probability", f"{not_success:.1f}%")
            st.progress(int(success))
            st.markdown('<div class="section-title">Why This Prediction?</div>', unsafe_allow_html=True)
            for _, r in get_contributions(input_df).iterrows():
                direction = "supports success" if r.Contribution > 0 else "supports not-successful"
                st.write(f"**{clean_feature_label(r.Feature)}** — {direction}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# PAGE 2
elif page == "Model Analysis":
    st.markdown('<div class="main-title">Model Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Understanding the performance and behavior of the selected machine-learning model.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Selected Model</div>', unsafe_allow_html=True)
    st.info("Logistic Regression was selected because it achieved the highest test accuracy among the evaluated models and provides interpretable feature contributions.")
    a,b,c,d = st.columns(4)
    a.metric("Accuracy", "85.68%")
    b.metric("Precision", "82.5%")
    c.metric("Recall", "77.9%")
    d.metric("F1 Score", "79.9%")

    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
    cm = np.array([[2355,157],[323,516]])
    fig, ax = plt.subplots(figsize=(6,4)); ax.imshow(cm)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Not Successful","Successful"]); ax.set_yticklabels(["Not Successful","Successful"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2): ax.text(j,i,cm[i,j],ha="center",va="center",fontsize=16)
    st.pyplot(fig)

    st.markdown('<div class="section-title">How the Model Works</div>', unsafe_allow_html=True)
    st.write("Game inputs are cleaned and transformed by the preprocessing pipeline. The processed features are passed to Logistic Regression, which calculates the probability of success and produces the final prediction.")
    st.markdown('<div class="section-title">Top Model Features</div>', unsafe_allow_html=True)
    if not feature_importance.empty and {"Feature","Coefficient"}.issubset(feature_importance.columns):
        fi = feature_importance.copy(); fi["Absolute"] = fi["Coefficient"].abs(); fi = fi.sort_values("Absolute", ascending=False).head(10)
        fi["Feature"] = fi["Feature"].apply(clean_feature_label)
        st.dataframe(fi[["Feature","Coefficient"]], use_container_width=True, hide_index=True)
    else:
        st.info("Run src/analyze_features.py to generate the feature importance file.")

# PAGE 3
elif page == "Model Comparison":
    st.markdown('<div class="main-title">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Comparison of the machine-learning algorithms evaluated during development.</div>', unsafe_allow_html=True)
    comparison = pd.DataFrame({"Model":["Logistic Regression","Random Forest","Tuned SVM","SVM"],"Accuracy":[85.68,83.02,82.99,81.86]})
    st.markdown('<div class="section-title">Test Accuracy</div>', unsafe_allow_html=True)
    st.bar_chart(comparison.set_index("Model"), y="Accuracy")
    st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
    shown = comparison.copy(); shown["Accuracy"] = shown["Accuracy"].map(lambda x:f"{x:.2f}%")
    st.dataframe(shown, use_container_width=True, hide_index=True)
    st.markdown('<div class="section-title">Why Logistic Regression?</div>', unsafe_allow_html=True)
    st.success("Logistic Regression achieved the highest test accuracy at 85.68% and was selected as the final model. Its coefficients also make individual predictions easier to explain.")

# PAGE 4
else:
    st.markdown('<div class="main-title">Dataset Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Visual insights from the Steam games used for model development.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Success Distribution</div>', unsafe_allow_html=True)
    counts = data["success"].value_counts().sort_index()
    st.bar_chart(pd.Series({"Not Successful":int(counts.get(0,0)),"Successful":int(counts.get(1,0))}))

    st.markdown('<div class="section-title">Games by Release Year</div>', unsafe_allow_html=True)
    st.line_chart(data.groupby("release_year").size())

    st.markdown('<div class="section-title">Success Rate by Release Era</div>', unsafe_allow_html=True)
    st.bar_chart(data.groupby("release_era")["success"].mean().mul(100))

    st.markdown('<div class="section-title">Average Price by Outcome</div>', unsafe_allow_html=True)
    price = data.groupby("success")["original_price"].mean().rename(index={0:"Not Successful",1:"Successful"})
    st.bar_chart(price)

    st.markdown('<div class="section-title">Average Achievements by Outcome</div>', unsafe_allow_html=True)
    achievements = data.groupby("success")["achievements"].mean().rename(index={0:"Not Successful",1:"Successful"})
    st.bar_chart(achievements)

    st.markdown('<div class="section-title">Success Rate by Genre</div>', unsafe_allow_html=True)
    genre_cols = [c for c in data.columns if c.startswith("genre_")]
    rows=[]
    for col in genre_cols:
        games=data[data[col]==1]
        if len(games)>=20: rows.append({"Genre":col.replace("genre_","").replace("_"," ").title(),"Success Rate":games.success.mean()*100})
    if rows:
        g=pd.DataFrame(rows).sort_values("Success Rate",ascending=False).set_index("Genre")
        st.bar_chart(g, y="Success Rate")
    st.info("These visualizations describe patterns in the dataset. They do not prove that a characteristic directly causes game success.")

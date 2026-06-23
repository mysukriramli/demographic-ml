import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import umap

# Set page layout to wide for side-by-side dashboard look
st.set_page_config(layout="wide", page_title="MyKad UMAP Classifier")

st.title("🧬 Live MyKad Demographics Classifier using UMAP")
st.markdown("""
    **Classroom Context:** Students submit a 5-digit continuous MyKad shorthand code. 
    The AI pipeline on the right standardizes their demographic features and applies **UMAP** to visually project and classify the entire room in real time!
""")

# --- Place of Birth (PB) Code Dictionary Mapping ---
pb_map = {
    "01": "Johor", "21": "Johor", "22": "Johor", "23": "Johor", "24": "Johor",
    "02": "Kedah", "25": "Kedah", "26": "Kedah", "27": "Kedah",
    "03": "Kelantan", "28": "Kelantan", "29": "Kelantan",
    "04": "Malacca", "30": "Malacca",
    "05": "Negeri Sembilan", "31": "Negeri Sembilan", "59": "Negeri Sembilan",
    "06": "Pahang", "32": "Pahang", "33": "Pahang",
    "07": "Penang", "34": "Penang", "35": "Penang",
    "08": "Perak", "36": "Perak", "37": "Perak", "38": "Perak", "39": "Perak",
    "09": "Perlis", "40": "Perlis",
    "10": "Selangor", "41": "Selangor", "42": "Selangor", "43": "Selangor", "44": "Selangor",
    "11": "Terengganu", "45": "Terengganu", "46": "Terengganu",
    "12": "Sabah", "47": "Sabah", "48": "Sabah", "49": "Sabah",
    "13": "Sarawak", "50": "Sarawak", "51": "Sarawak", "52": "Sarawak", "53": "Sarawak",
    "14": "Kuala Lumpur", "54": "Kuala Lumpur", "55": "Kuala Lumpur", "56": "Kuala Lumpur", "57": "Kuala Lumpur",
    "15": "Labuan", "58": "Labuan", "16": "Putrajaya"
}

# --- GLOBAL SHARED DATABASE MANAGER ---
class DemographicDataManager:
    def __init__(self):
        # Pre-populate with diverse dummy data so UMAP has background clusters immediately
        np.random.seed(42)
        dummy_traders = [f"Agent_{i:02d}" for i in range(1, 31)]
        dummy_years = np.random.choice(["88", "95", "99", "01", "05", "15"], size=30)
        dummy_pbs = np.random.choice(["01", "02", "03", "08", "10", "14"], size=30)
        dummy_gs = np.random.choice(["1", "2", "3", "4"], size=30)
        
        records = []
        for t, y, p, g in zip(dummy_traders, dummy_years, dummy_pbs, dummy_gs):
            records.append({"Alias": t, "YY": y, "PB": p, "G": g})
            
        self.df = pd.DataFrame(records)
    
    def add_profile(self, row):
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        
    def reset_database(self):
        self.df = pd.DataFrame(columns=["Alias", "YY", "PB", "G"])

@st.cache_resource
def get_demographic_database():
    return DemographicDataManager()

db = get_demographic_database()

# Layout Split: 1/3 Submission Input, 2/3 Live UMAP Charts
col1, col2 = st.columns([1, 2])

# --- COLUMN 1: CONTINUOUS SHORTHAND INPUT FORM ---
with col1:
    st.header("📥 Continuous Data Stream")
    
    # Simple instructions for seamless continuous entry
    st.markdown("💡 *Type your 5 digits continuously below without lifting your finger.*")
    
    user_name = st.text_input("Enter your Shorthand Alias:", placeholder="e.g., Participant X", key="alias_input")
    user_code = st.text_input("Enter 5 Digits Continuously (YYPBG):", max_chars=5, placeholder="e.g., 99101", key="code_input")
    
    if len(user_code) == 5 and user_code.isdigit():
        if user_name.strip() == "":
            st.error("Please provide an Alias name above your continuous code entry!")
        else:
            # Check if this precise entry was already added to prevent infinite re-run loops
            is_duplicate = ((db.df["Alias"] == user_name) & (db.df["YY"] == user_code[0:2])).any()
            if not is_duplicate:
                new_profile = {
                    "Alias": user_name,
                    "YY": user_code[0:2],
                    "PB": user_code[2:4],
                    "G": user_code[4]
                }
                db.add_profile(new_profile)
                st.success(f"⚡ Live Profile compiled successfully for {user_name}!")
    elif len(user_code) > 0 and not user_code.isdigit():
        st.error("Invalid character vector detected. Numerical parameters only.")

# --- COLUMN 2: REAL-TIME CLASSIFICATION & UMAP PLOT ---
with col2:
    st.header("🖥️ Live UMAP Analytics Engine")
    
    if st.button("🔄 Sync Class Matrix", type="primary", use_container_width=True):
        st.rerun()
        
    df = db.df.copy()
    
    if len(df) >= 5:
        # --- FEATURE ENGINEERING PIPELINE ---
        # 1. Resolve true birth year & age (Current year: 2026)
        df["YY_int"] = df["YY"].astype(int)
        df["Birth Year"] = df["YY_int"].apply(lambda x: 2000 + x if x <= 26 else 1900 + x)
        df["Age"] = 2026 - df["Birth Year"]
        
        # 2. Parse State Names and Categorical variables
        df["State Group"] = df["PB"].map(pb_map).fillna("Other / Unknown")
        df["Gender Group"] = df["G"].astype(int).apply(lambda x: "Female" if x % 2 == 0 else "Male")
        
        # 3. Formulate pure numeric metrics array for UMAP mathematical computation
        df["PB_Numeric"] = df["PB"].astype(int)
        df["G_Numeric"] = df["G"].astype(int)
        
        umap_features = df[["Age", "PB_Numeric", "G_Numeric"]]
        
        # Scale feature space inputs equally
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(umap_features)
        
        # Run UMAP Dimensionality Reduction
        # Adaptive adjustment of neighbors based on the database size
        n_neighbors = min(15, len(df) - 1)
        if n_neighbors < 2:
            n_neighbors = 2
            
        try:
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
            embedding = reducer.fit_transform(scaled_features)
            
            df["UMAP Dim 1"] = embedding[:, 0]
            df["UMAP Dim 2"] = embedding[:, 1]
            
            # --- RENDER INTERACTIVE PLOTLY SCATTER CHART ---
            color_option = st.radio("Color Map Classification Field:", ["State Group", "Gender Group", "Birth Year"], horizontal=True)
            
            fig = px.scatter(
                df, 
                x="UMAP Dim 1", 
                y="UMAP Dim 2", 
                color=color_option,
                hover_name="Alias",
                hover_data=["Age", "State Group", "Gender Group"],
                title=f"2D UMAP Spatial Embedding Topology (Grouped by {color_option})",
                template="plotly_white"
            )
            fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color="DarkSlateGrey")))
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning("UMAP mathematical optimization loop is converging. Submit more distinct entries.")
            
        # Display underlying registry log metrics
        st.subheader("📋 Registry Profile Metrics Logs")
        st.dataframe(df[["Alias", "Age", "State Group", "Gender Group"]], use_container_width=True)
    else:
        st.info("Awaiting cluster density parameters. Please feed more values into the data stream.")

# --- CAMOUFLAGED TEACHER RESET KEY ---
st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
with st.expander("·", expanded=False):
    if st.button("🗑️ Wipe Model Cache (Clear All Class Entries)"):
        db.reset_database()
        st.rerun()

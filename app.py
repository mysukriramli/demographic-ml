import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import umap
from st_keyup import st_keyup

# Set page layout to wide for dashboard split look
st.set_page_config(layout="wide", page_title="MyKad Masked UMAP Engine")

# Inject Custom CSS to style input elements and the giant fill-in-the-blank mask
st.markdown("""
    <style>
        div[data-baseweb="input"] input {
            font-size: 32px !important;
            height: 65px !important;
            text-align: center !important;
            font-family: 'Courier New', monospace !important;
            font-weight: bold !important;
            letter-spacing: 2px !important;
        }
        .giant-mask {
            font-size: 54px !important;
            font-family: 'Courier New', monospace !important;
            font-weight: bold !important;
            color: #1E40AF !important;
            text-align: center !important;
            letter-spacing: 6px !important;
            background-color: #F3F4F6 !important;
            padding: 20px !important;
            border-radius: 12px !important;
            border: 3px solid #3B82F6 !important;
            margin-bottom: 25px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🚨 Anonymized Identity Demographics Classifier")
st.markdown("""
    **Classroom Context:** Students type only their Birth Year (YY), Birth State Code (PB), and Gender Digit (G) 
    continuously as a 5-digit stream. The system automatically jumps over the private parameters using a secure mask!
""")

# --- Safe Place of Birth (PB) Mapping Base ---
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
class CloudRegistryManager:
    def __init__(self):
        self.df = pd.DataFrame(columns=["Raw_Vector", "YY", "PB", "G"])
    
    def add_entry(self, raw_vector, yy, pb, g):
        rv_str = str(raw_vector).strip()
        if rv_str and not (self.df["Raw_Vector"] == rv_str).any():
            new_row = {"Raw_Vector": rv_str, "YY": str(yy), "PB": str(pb), "G": str(g)}
            self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
            
    def wipe_all(self):
        self.df = pd.DataFrame(columns=["Raw_Vector", "YY", "PB", "G"])

@st.cache_resource
def get_cloud_registry():
    return CloudRegistryManager()

db = get_cloud_registry()

# Layout Split: 1/3 Guided Input Column, 2/3 Live Visual Dashboard
col1, col2 = st.columns([1, 2])

# --- COLUMN 1: INSTANT TYPING PORTAL WITH EXPLICIT SEND BUTTON ---
with col1:
    st.header("📥 Continuous Entry")
    
    st.markdown("### 📲 Scan to Join Live")
    app_url = "https://demographic-ml-37tvhybmcrlgumshlapppp4o.streamlit.app/"
    qr_api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=180x180&data={app_url}"
    st.image(qr_api_url, caption="Scan with your phone to type live", width=180)
    st.markdown("---")
    
    # Natively processes keystrokes live on every stroke
    live_input = st_keyup(
        label="Type continuous stream: YY then PB then G",
        value="",
        max_chars=5,
        key="live_key_stream"
    )
    
    # Filter numbers out safely
    clean_digits = "".join([c for c in live_input if c.isdigit()])[0:5]
    
    # --- LIVE FILL-IN-THE-BLANK MASK CONTROLLER ---
    yy_part = clean_digits[0:2]
    pb_part = clean_digits[2:4]
    g_part = clean_digits[4:5]
    
    disp_yy = yy_part + "_" * (2 - len(yy_part))
    disp_pb = pb_part + "_" * (2 - len(pb_part)) if len(yy_part) == 2 else "__"
    disp_g = g_part if len(clean_digits) == 5 else "_"
    
    # Renders the precise fill-in-the-blank text mask format on screen
    current_mask_view = f"{disp_yy}####-{disp_pb}-###{disp_g}"
    st.markdown(f"<div class='giant-mask'>{current_mask_view}</div>", unsafe_allow_html=True)
    
    # --- THE SEND CONTROL BUTTON ---
    send_button = st.button("🚀 Send Profile to Matrix", type="primary", use_container_width=True)
    
    if send_button:
        if len(clean_digits) == 5:
            db.add_entry(clean_digits, yy_part, pb_part, g_part)
            st.success("🎯 Pattern linked successfully! Check the main projector screen.")
        else:
            st.error(f"❌ Complete the pattern first! Missing digits: {5 - len(clean_digits)}/5 remaining.")
            
    if len(clean_digits) > 0 and len(clean_digits) < 5:
        st.info(f"Typing progress: {len(clean_digits)} / 5 digits tracked.")

# --- COLUMN 2: LIVE UMAP CLASSIFIER PLATFORM ---
with col2:
    st.header("🖥️ Live UMAP Projection Center")
    
    if st.button("🔄 Sync Classroom Network Matrix", use_container_width=True):
        st.rerun()
        
    df = db.df.copy()
    total_records = len(df)
    
    st.metric(label="Total Anonymous Profiles Submitted", value=total_records)
    
    # Run UMAP once a basic data cluster size is achieved
    if total_records >= 4:
        df["YY_int"] = df["YY"].astype(int)
        df["PB_int"] = df["PB"].astype(int)
        df["G_int"] = df["G"].astype(int)
        
        # Calculate generation age vectors relative to 2026 boundary conditions
        df["Birth Year"] = df["YY_int"].apply(lambda x: 2000 + x if x <= 26 else 1900 + x)
        df["Age"] = 2026 - df["Birth Year"]
        
        # Safe fallback method dictionary extraction to prevent any KeyError
        df["Birth State"] = df["PB"].apply(lambda x: pb_map.get(str(x), "International / Other"))
        df["Assigned Gender"] = df["G_int"].apply(lambda x: "Female" if x % 2 == 0 else "Male")
        
        df["Anonymized Label"] = df.index.map(lambda x: f"Identity_{x+1:02d}")
        
        processing_matrix = df[["Age", "PB_int", "G_int"]]
        
        scaler = StandardScaler()
        scaled_matrix = scaler.fit_transform(processing_matrix)
        
        neighbors_calc = min(15, total_records - 1)
        if neighbors_calc < 2: 
            neighbors_calc = 2
            
        try:
            reducer = umap.UMAP(n_neighbors=neighbors_calc, min_dist=0.1, random_state=42)
            embedding = reducer.fit_transform(scaled_matrix)
            
            df["UMAP Axis 1"] = embedding[:, 0]
            df["UMAP Axis 2"] = embedding[:, 1]
            
            classification_target = st.radio("Classify Topology Coordinates By:", ["Birth State", "Assigned Gender", "Birth Year"], horizontal=True)
            
            # 1. Base scatter plot layout
            fig = px.scatter(
                df,
                x="UMAP Axis 1",
                y="UMAP Axis 2",
                color=classification_target,
                hover_name="Anonymized Label",
                hover_data=["Age", "Birth State", "Assigned Gender"],
                template="plotly_white"
            )
            
            # 2. --- DRAW CLOUD AROUND CLUSTERS (Density Contour Overlay) ---
            # Generates mathematical boundaries enclosing point masses dynamically
            fig_contour = px.density_contour(df, x="UMAP Axis 1", y="UMAP Axis 2")
            for trace in fig_contour.data:
                trace.update(
                    line=dict(color="rgba(59, 130, 246, 0.35)", width=2, dash="dashdot"),
                    contours=dict(coloring="none")
                )
                fig.add_trace(trace)
                
            fig.update_traces(marker=dict(size=14, opacity=0.9, line=dict(width=1, color="DarkSlateGrey")), selector=dict(type='scatter'))
            fig.update_layout(title=f"Live Spatial Topology with Cluster Density Clouds (Grouped by {classification_target})")
            st.plotly_chart(fig, use_container_width=True)
            
            # --- 🤖 AUTOMATED DATA INSIGHTS ENGINE (Auto-Commentary) ---
            st.subheader("🤖 Automated Demographic Intelligence Report")
            
            top_state = df["Birth State"].value_counts().idxmax()
            state_count = df["Birth State"].value_counts().max()
            male_percentage = (df["Assigned Gender"] == "Male").sum() / total_records * 100
            avg_classroom_age = df["Age"].mean()
            
            # Compile a dynamic narrative summary based on real-time matrix changes
            st.markdown(f"""
            * **Spatial Core Composition:** The cluster space currently identifies a high-density node centered around individuals born in **{top_state}** (accounting for {state_count} submitted profiles).
            * **Geometric Gender Distribution:** The mathematical topology is split across a **{male_percentage:.1f}% Male** vector alignment. The spatial distances on the graph show how clearly UMAP separates these binary code values.
            * **Generational Variance:** The average historical age vector of this room resolves to **{avg_classroom_age:.1f} years old**. Points tracking closely together along the vertical axis represent shared generation layers.
            """)
            
        except Exception as e:
            st.info("Gathering structural point variations... Submit a few more distinct entries to settle spatial vectors and build the cluster clouds.")
            
        st.subheader("📋 Current Registry Overview")
        st.dataframe(df[["Anonymized Label", "Birth Year", "Birth State", "Assigned Gender"]], use_container_width=True)
        
    else:
        st.info("Awaiting cluster density validation. Please submit at least **4 unique records** from student phones to activate the UMAP neural pipeline.")

# --- SECURE CAMOUFLAGED CLEANSE SYSTEM ---
st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
with st.expander("·", expanded=False):
    if st.button("🗑️ Clear Live Database Cache"):
        db.wipe_all()
        st.rerun()

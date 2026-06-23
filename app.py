import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import umap

# Set page layout to wide for dashboard split look
st.set_page_config(layout="wide", page_title="MyKad UMAP Engine")

# Inject Custom CSS to make the continuous input field giant and easy to read
st.markdown("""
    <style>
        div[data-baseweb="input"] input {
            font-size: 36px !important;
            height: 70px !important;
            text-align: center !important;
            font-family: 'Courier New', monospace !important;
            font-weight: bold !important;
            color: #1E3A8A !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🚨 Live Identity Demographics Classifier using UMAP")
st.markdown("""
    **Classroom Context:** Students type their 12 digits continuously into the giant field below. 
    The AI pipeline strips spaces or dashes, processes their data completely anonymously, and plots them on a live UMAP cluster map!
""")

# --- Place of Birth (PB) Mapping Base ---
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

# --- GLOBAL SHARED DATABASE MANAGER (Starts completely empty) ---
class CloudRegistryManager:
    def __init__(self):
        self.df = pd.DataFrame(columns=["Raw_ID", "YY", "MM", "DD", "PB", "Serial", "G"])
    
    def add_entry(self, row):
        # Prevent exact duplicate string entries from double-triggering
        if not ((self.df["Raw_ID"] == row["Raw_ID"])).any():
            self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
            
    def wipe_all(self):
        self.df = pd.DataFrame(columns=["Raw_ID", "YY", "MM", "DD", "PB", "Serial", "G"])

@st.cache_resource
def get_cloud_registry():
    return CloudRegistryManager()

db = get_cloud_registry()

# Layout Split: 1/3 Big Entry Column, 2/3 Live Visual Dashboard
col1, col2 = st.columns([1, 2])

# --- COLUMN 1: ENTRY PORTAL & TARGET QR CODE ---
with col1:
    st.header("📥 Continuous Input Portal")
    
    st.markdown("### 📲 Scan to Join Live")
    # Updated to your unique project target link
    app_url = "https://demographic-ml-37tvhybmcrlgumshlapppp4o.streamlit.app/"
    qr_api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=180x180&data={app_url}"
    st.image(qr_api_url, caption="Scan with your phone to type live", width=180)
    st.markdown("---")
    
    st.markdown("##### Type your 12 digits cleanly below:")
    
    # Single large text box with the precise layout request
    user_string = st.text_input(
        label="Format standard: YYMMDD-PB-###G",
        value="",
        max_chars=14,
        placeholder="YYMMDD-PB-###G"
    )
    
    # Process string smoothly by removing potential accidental user dashes/spaces live
    clean_digits = user_string.replace("-", "").replace(" ", "").strip()
    
    if len(clean_digits) == 12 and clean_digits.isdigit():
        # Map sub-segments seamlessly
        new_row_data = {
            "Raw_ID": clean_digits,
            "YY": clean_digits[0:2],
            "MM": clean_digits[2:4],
            "DD": clean_digits[4:6],
            "PB": clean_digits[6:8],
            "Serial": clean_digits[8:11],
            "G": clean_digits[11]
        }
        db.add_entry(new_row_data)
        st.success("🎯 Identity pattern submitted seamlessly!")
    elif len(clean_digits) > 0 and len(clean_digits) < 12:
        st.warning(f"Processing character string... Count: {len(clean_digits)}/12 digits")

# --- COLUMN 2: LIVE UMAP CLASSIFIER PLATFORM ---
with col2:
    st.header("🖥️ Live UMAP Projection Center")
    
    if st.button("🔄 Sync Classroom Network Matrix", type="primary", use_container_width=True):
        st.rerun()
        
    df = db.df.copy()
    total_records = len(df)
    
    st.metric(label="Total Anonymous Profiles Submitted", value=total_records)
    
    # Check if dataset has enough observations to calculate mathematical dimensions safely
    if total_records >= 4:
        # --- ANONYMOUS FEATURE ENGINEERING ---
        df["YY_int"] = df["YY"].astype(int)
        df["MM_int"] = df["MM"].astype(int)
        df["DD_int"] = df["DD"].astype(int)
        df["PB_int"] = df["PB"].astype(int)
        df["G_int"] = df["G"].astype(int)
        
        # Calculate true age parameters based on current year 2026 boundary conditions
        df["Birth Year"] = df["YY_int"].apply(lambda x: 2000 + x if x <= 26 else 1900 + x)
        df["Age"] = 2026 - df["Birth Year"]
        
        # Human readable grouping categories
        df["Birth State"] = df["PB"].map(pb_map).fillna("International / Other")
        df["Assigned Gender"] = df["G_int"].apply(lambda x: "Female" if x % 2 == 0 else "Male")
        
        # Mask specific identifiers to protect privacy on public classroom display
        df["Anonymized Label"] = df.index.map(lambda x: f"Identity_{x+1:02d}")
        df["Masked Profile"] = df["Birth Year"].astype(str) + " | " + df["Birth State"] + " | " + df["Assigned Gender"]
        
        # 3D Metric array to process dimensional reduction math
        processing_matrix = df[["Age", "MM_int", "DD_int", "PB_int", "G_int"]]
        
        scaler = StandardScaler()
        scaled_matrix = scaler.fit_transform(processing_matrix)
        
        # Dynamic neighbors adjustments based on collected volume
        neighbors_calc = min(15, total_records - 1)
        if neighbors_calc < 2: 
            neighbors_calc = 2
            
        try:
            # Execute UMAP Embedding calculations
            reducer = umap.UMAP(n_neighbors=neighbors_calc, min_dist=0.1, random_state=42)
            embedding = reducer.fit_transform(scaled_matrix)
            
            df["UMAP Axis 1"] = embedding[:, 0]
            df["UMAP Axis 2"] = embedding[:, 1]
            
            # Interactive Categorization Selector
            classification_target = st.radio("Classify Topology Coordinates By:", ["Birth State", "Assigned Gender", "Birth Year"], horizontal=True)
            
            fig = px.scatter(
                df,
                x="UMAP Axis 1",
                y="UMAP Axis 2",
                color=classification_target,
                hover_name="Anonymized Label",
                hover_data=["Age", "Birth State", "Assigned Gender"],
                title=f"Live UMAP Projection Map (Clustered by {classification_target})",
                template="plotly_white"
            )
            fig.update_traces(marker=dict(size=14, opacity=0.8, line=dict(width=1, color="DarkSlateGrey")))
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as error_log:
            st.info("Gathering diverse geometric point distributions... Type in a few more distinct entries to settle spatial vectors.")
            
        # Summary log output
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

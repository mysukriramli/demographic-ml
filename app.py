import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import umap
from st_keyup import st_keyup

# Set page layout to wide for dashboard split look
st.set_page_config(layout="wide", page_title="MyKad Automated UMAP Engine")

# Inject Custom CSS to style the input box and make the fill-in-the-blank mask giant
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
    
    current_mask_view = f"{disp_yy}####-{disp_pb}-###{disp_g}"
    st.markdown(f"<div class='giant-mask'>{current_mask_view}</div>", unsafe_allow_html=True)
    
    # Send Control Button
    send_button = st.button("🚀 Send Profile to Matrix", type="primary", use_container_width=True)
    
    if send_button:
        if len(clean_digits) == 5:
            db.add_entry(clean_digits, yy_part, pb_part, g_part)
            st.success("🎯 Pattern linked successfully! Check the main projector screen.")
        else:
            st.error(f"❌ Complete the pattern first! Missing digits: {5 - len(clean_digits)}/5 remaining.")
            
    if len(clean_digits) > 0 and len(clean_digits) < 5:
        st.info(f"Typing progress: {len(clean_digits)} / 5 digits tracked.")

# --- COLUMN 2: LIVE UMAP CLASSIFIER & AUTOMATED COMMENTARY ---
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
            
        except Exception as e:
            st.info("Gathering structural point variations... Submit a few more distinct entries to settle spatial vectors.")
            
        # --- NEW SECTION: AUTOMATED DEMOGRAPHIC COMMENTARY REPORT ---
        st.markdown("---")
        st.subheader("🤖 Automated Demographic Intelligence Report")
        
        # 1. Geographic Concentration Commentary
        top_state = df["Birth State"].mode()[0]
        state_counts = df["Birth State"].value_value_counts = df["Birth State"].value_counts()
        top_state_pct = (state_counts.iloc[0] / total_records) * 100
        
        # 2. Generational Commentary (Relative to 2026)
        gen_z_count = len(df[(df["Birth Year"] >= 1997) & (df["Birth Year"] <= 2009)])
        gen_alpha_count = len(df[df["Birth Year"] >= 2010])
        millennial_count = len(df[(df["Birth Year"] >= 1981) & (df["Birth Year"] <= 1996)])
        
        # 3. Gender Matrix Commentary
        male_count = len(df[df["Assigned Gender"] == "Male"])
        female_count = len(df[df["Assigned Gender"] == "Female"])
        
        # Render dynamic analytical summary cards based on incoming matrix trends
        st.markdown(f"""
        ### 📊 Spatial Clustering Analysis:
        * **Geographic Hotspot:** The most heavily represented region in the current dataset is **{top_state}**, which accounts for **{top_state_pct:.1f}%** of all entries. On the UMAP chart, this forms a dense coordinate cluster due to shared geographic code values.
        * **Generational Variance:** In our active 2026 data stream, the class matrix is structurally divided into **{gen_alpha_count}** Gen Alpha profiles, **{gen_z_count}** Gen Z profiles, and **{millennial_count}** Millennial profiles. UMAP utilizes these age vectors to spread data points smoothly along the age axis.
        * **Gender Matrix Ratios:** The current room balance stands at **{male_count} Male entries** vs. **{female_count} Female entries**. When toggling the color map to 'Assigned Gender', you will notice a distinct, clean mathematical bisection on the chart because the final odd/even digits create a highly polarized spatial separation.
        """)
        
        if total_records >= 10:
            st.success("💡 **Data Science Insight:** With over 10 profiles mapped, notice how UMAP's non-linear manifold math automatically groups people who share the exact same state and gender close together, while cleanly separating different age generations into distinct bands!")
            
        # Summary log output table
        st.subheader("📋 Current Registry Overview")
        st.dataframe(df[["Anonymized Label", "Birth Year", "Birth State", "Assigned Gender"]], use_container_width=True)
        
    else:
        st.info("Awaiting cluster density validation. Please submit at least **4 unique records** from student phones to activate the UMAP neural pipeline and trigger auto-commentary.")

# --- SECURE CAMOUFLAGED CLEANSE SYSTEM ---
st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
with st.expander("·", expanded=False):
    if st.button("🗑️ Clear Live Database Cache"):
        db.wipe_all()
        st.rerun()

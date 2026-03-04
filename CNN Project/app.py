import streamlit as st
import pandas as pd
import os
import requests
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from models.train import start_training
from models.predict import make_prediction

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Neural Studio Pro 2026", layout="wide", page_icon="🧠")

# --- 2. ADVANCED NEON & GLASS UI ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0e1117 0%, #07090d 100%); color: #ffffff; }
    
    /* Neon Text Effect */
    .neon-header {
        font-size: 45px; font-weight: bold; color: #fff;
        text-shadow: 0 0 10px #00bcd4, 0 0 20px #00bcd4;
        text-align: center; margin-bottom: 30px;
        font-family: 'Orbitron', sans-serif;
    }

    /* Glassmorphism Cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        padding: 25px; border-radius: 20px;
        border: 1px solid rgba(0, 188, 212, 0.3);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] { background-color: #0a0a12 !important; border-right: 1px solid #1f1f2e; }
    
    /* Custom Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #00bcd4, #007c91);
        color: white; border: none; border-radius: 10px;
        padding: 12px; font-weight: bold; transition: 0.3s;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0, 188, 212, 0.4); }
    </style>
    """, unsafe_allow_html=True)

# --- 3. ASSETS LOADING (With Safety) ---
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except: return None

lottie_dash = load_lottieurl("https://lottie.host/804044b7-d95a-4b95-8a9d-598d9c5784f1/f6mKjJtXIn.json")
lottie_train = load_lottieurl("https://lottie.host/67705190-2810-444f-a89a-4122822a969a/Y8V9m1S2vY.json")

# --- 4. DIRECTORY SETUP ---
BASE_DATA_PATH = "Dataset"
if not os.path.exists(BASE_DATA_PATH): os.makedirs(BASE_DATA_PATH)

# --- 5. MODERN SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #00bcd4; font-size: 25px;'>💎 NEURAL ENGINE</h1>", unsafe_allow_html=True)
    st.markdown("---")
    menu = option_menu(
        menu_title=None,
        options=["Dashboard", "Data Collector", "Train Model", "Prediction"],
        icons=["grid-fill", "cloud-arrow-up-fill", "cpu-fill", "search-heart-fill"],
        menu_icon="cast", default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#00bcd4", "font-size": "20px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"5px", "color": "white"},
            "nav-link-selected": {"background-color": "rgba(0, 188, 212, 0.2)", "border-left": "4px solid #00bcd4"},
        }
    )

# --- 6. PAGE LOGIC ---

# --- DASHBOARD ---
if menu == "Dashboard":
    st.markdown('<p class="neon-header">SYSTEM ANALYTICS</p>', unsafe_allow_html=True)
    if lottie_dash: st_lottie(lottie_dash, height=150)
    
    classes = [d for d in os.listdir(BASE_DATA_PATH) if os.path.isdir(os.path.join(BASE_DATA_PATH, d))]
    
    if not classes:
        st.info("⚠️ Engine is empty. Please go to 'Data Collector' to add classes.")
    else:
        total_imgs = sum([len(os.listdir(os.path.join(BASE_DATA_PATH, c))) for c in classes])
        
        # 1. Top Metrics
        col1, col2 = st.columns(2)
        col1.metric("Active Categories", len(classes))
        col2.metric("Total Neural Samples", total_imgs)
        st.markdown("---")
        
        # 2. Donut Chart & Cards Layout
        st.markdown("### 📊 Dataset Distribution")
        chart_col, grid_col = st.columns([1, 1])
        
        with chart_col:
            # Beautiful Plotly Donut Chart
            class_counts = [len(os.listdir(os.path.join(BASE_DATA_PATH, c))) for c in classes]
            fig = go.Figure(data=[go.Pie(labels=classes, values=class_counts, hole=.5, 
                                         marker_colors=['#00bcd4', '#ff007f', '#00e676', '#ffea00', '#2979ff'])])
            fig.update_layout(template="plotly_dark", margin=dict(t=30, b=10, l=10, r=10), 
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        with grid_col:
            # Classes in Grid format (Cards)
            st.markdown("### 📁 Manage Classes")
            cols = st.columns(2) # 2 columns grid
            for i, c in enumerate(classes):
                img_count = len(os.listdir(os.path.join(BASE_DATA_PATH, c)))
                with cols[i % 2]:
                    st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; 
                                    border-left: 3px solid #00bcd4; margin-bottom: 15px;">
                            <h4 style="margin:0; color:#ffffff;">{c}</h4>
                            <p style="margin:0; color:#aaaaaa;">Samples: <b style="color:#00bcd4;">{img_count}</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                    if st.button("🗑️ Delete", key=f"del_{c}", help=f"Delete {c} class"):
                        import shutil
                        shutil.rmtree(os.path.join(BASE_DATA_PATH, c))
                        st.rerun()

# DATA COLLECTOR
# --- DATA COLLECTOR ---
elif menu == "Data Collector":
    st.markdown('<p class="neon-header">DATA ACQUISITION ENGINE</p>', unsafe_allow_html=True)
    
    # VIP Info Box
    st.markdown("""
        <div style="background: rgba(0, 188, 212, 0.1); border-left: 4px solid #00bcd4; padding: 15px; border-radius: 8px; margin-bottom: 25px;">
            <h4 style="color: #00bcd4; margin-top: 0;">💡 Pro Tip for Maximum Accuracy:</h4>
            <p style="color: #e0e0e0; margin-bottom: 0;">Har class ke liye kam se kam <b>30-50 images</b> upload karein. Jitna zyada data, utna smart AI!</p>
        </div>
    """, unsafe_allow_html=True)
    
    # 2-Column Layout for better look
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("### 🏷️ 1. Define Identity")
        c_name = st.text_input("Class Name", placeholder="e.g. Tiger, Car, Apple...", label_visibility="collapsed")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### ⚙️ 3. Initialize Sync")
        upload_btn = st.button("🚀 UPLOAD TO NEURAL CORE", use_container_width=True)

    with col2:
        st.markdown("### 📥 2. Drop Neural Samples")
        files = st.file_uploader("Upload Images", accept_multiple_files=True, label_visibility="collapsed")
    
    st.markdown("---")
    
    # Action Logic with Progress Bar
    if upload_btn:
        if c_name and files:
            # Name ko clean karna (taki spacing ya capital letters ka masla na ho)
            clean_name = c_name.strip().lower()
            path = os.path.join(BASE_DATA_PATH, clean_name)
            if not os.path.exists(path): 
                os.makedirs(path)
            
            # Cool Progress Bar Animation
            progress_text = f"Encrypting and Storing {len(files)} samples..."
            my_bar = st.progress(0, text=progress_text)
            
            for i, f in enumerate(files):
                with open(os.path.join(path, f.name), "wb") as file: 
                    file.write(f.getbuffer())
                # Update progress bar
                my_bar.progress((i + 1) / len(files), text=progress_text)
            
            st.success(f"✅ Successfully ingested {len(files)} samples into '{clean_name}' category!")
            st.toast("Data Synced to Core!", icon="🔥")
            st.balloons()
            
        else: 
            st.error("⚠️ System Alert: Please provide both an Identity (Name) and Neural Samples (Images).")

# TRAINING
# --- MODEL TRAINING (CLEAN & PROFESSIONAL VERSION) ---
elif menu == "Train Model":
    st.markdown('<p class="neon-header">NEURAL CORE TRAINING</p>', unsafe_allow_html=True)
    
    col_anim, col_info = st.columns([1, 1.5])
    
    with col_anim:
        if lottie_train: 
            st_lottie(lottie_train, height=250)
    
    with col_info:
        st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 15px; border: 1px solid #00bcd4;">
                <h3 style="color: #00bcd4; margin-top:0;">🧠 Intelligence Configuration</h3>
                <p>Adjust the <b>Iteration Intensity (Epochs)</b>. Higher values allow the neural network to learn deeper features from your dataset.</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        epochs = st.select_slider("Select Training Intensity", options=[5, 10, 20, 30, 50], value=10)
        train_btn = st.button("🚀 EXECUTE NEURAL LEARNING", use_container_width=True)

    st.markdown("---")

    if train_btn:
        classes = [d for d in os.listdir(BASE_DATA_PATH) if os.path.isdir(os.path.join(BASE_DATA_PATH, d))]
        if len(classes) < 2:
            st.error("❌ System Alert: Minimum 2 classes required to initialize training.")
        else:
            # 1. Neural Console Log
            st.markdown("### 💻 Neural Console Log")
            with st.status("Initializing Neural Layers...", expanded=True) as status:
                st.write("🔗 Establishing connections to Dataset...")
                st.write(f"📂 Found {len(classes)} distinct classes.")
                st.write("⚙️ Optimizing CNN Architecture...")
                
                # Actual Training Call
                history, cm, class_indices = start_training(BASE_DATA_PATH, epochs=epochs)
                
                st.write("✅ Weights saved to: `saved_models/model.h5`")
                status.update(label="Training Sequence Complete!", state="complete", expanded=False)

            # --- CLEAN FEEDBACK (No Balloons, No Snow) ---
            st.toast('System Status: Neural Core Optimized', icon='✅')
            st.success("Analysis Complete: Neural Model successfully trained and weights stored.")

            st.markdown("<br>", unsafe_allow_html=True)
            
            # 2. Results Metrics
            res1, res2, res3 = st.columns(3)
            final_acc = history.history['accuracy'][-1] * 100
            final_loss = history.history['loss'][-1]
            
            res1.metric("Final Accuracy", f"{final_acc:.2f}%")
            res2.metric("Final Loss", f"{final_loss:.4f}")
            res3.metric("Status", "OPTIMIZED")

            # 3. Visual Analytics (Tabs Layout)
            tab1, tab2 = st.tabs(["📈 Learning Curves", "🎯 Evaluation Matrix"])
            
            with tab1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=history.history['accuracy'], name="Train Accuracy", line=dict(color='#00bcd4', width=3)))
                fig.add_trace(go.Scatter(y=history.history['val_accuracy'], name="Val Accuracy", line=dict(color='#ff007f', dash='dash')))
                fig.update_layout(template="plotly_dark", title="Learning Progression", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                class_names = list(class_indices.keys())
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm, x=class_names, y=class_names,
                    colorscale='Viridis',
                    texttemplate="%{z}",
                    showscale=True
                ))
                fig_cm.update_layout(title="Confusion Matrix: Predicted vs True Labels", template="plotly_dark")
                st.plotly_chart(fig_cm, use_container_width=True)
# PREDICTION
elif menu == "Prediction":
    st.markdown('<p class="neon-header">LIVE INFERENCE</p>', unsafe_allow_html=True)
    if not os.path.exists("saved_models/model.h5"):
        st.error("Engine Offline. Train the model first.")
    else:
        test_file = st.file_uploader("Upload Image for Inference")
        if test_file:
            st.image(test_file, width=300)
            if st.button("IDENTIFY"):
                with open("temp.jpg", "wb") as f: f.write(test_file.getbuffer())
                label, conf = make_prediction("temp.jpg")
                st.markdown(f"""
                    <div style="border: 2px solid #00bcd4; padding: 20px; border-radius: 15px; text-align: center;">
                        <h2 style="color: #00bcd4;">IDENTITY: {label}</h2>
                        <p>Confidence: {conf:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()
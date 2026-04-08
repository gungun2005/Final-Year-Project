import streamlit as st
from ai_logic import analyze_leaf
from nlp_logic import generate_treatment_plan
from datetime import datetime
from gtts import gTTS
import os
import time

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="AI Crop Doctor", 
    page_icon="🌿", 
    layout="wide", 
    initial_sidebar_state="expanded" 
)

# 2. SESSION STATE
if 'history' not in st.session_state:
    st.session_state.history = []

# 3. LIGHTWEIGHT CSS (Only for button polish, safely avoiding menus)
st.markdown("""
<style>
    div.stButton > button {
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# 4. THE SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1892/1892747.png", width=60) # Adds a clean little logo
    st.title("History Log")
    st.divider()
    
    if not st.session_state.history:
        st.info("No scans performed yet.")
    else:
        for item in reversed(st.session_state.history):
            st.markdown(f"**🕒 {item['time']}**")
            st.caption(f"🧬 {item['disease']}")
            st.divider()

# 5. DASHBOARD HEADER
st.title("🌿 AI Crop Doctor")
st.markdown("### Professional Disease Detection & Treatment Intelligence")
st.divider()

# 6. MAIN UI: SPLIT DASHBOARD LAYOUT
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("1. Visual Input")
    st.info("Upload a clear image of the affected plant leaf.")
    uploaded_image = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_image:
        st.image(uploaded_image, caption="Image Processed Successfully", use_container_width=True)

with col2:
    st.subheader("2. Audio & Context")
    st.info("Select the farmer's dialect and provide specific context.")
    
    languages = ["English", "Hindi", "Bengali", "Telugu", "Marathi", "Tamil", "Urdu", "Gujarati", "Kannada", "Malayalam", "Odia", "Punjabi", "Assamese"]
    selected_lang = st.selectbox("Translation Language", languages)
    
    user_question = st.text_input("Context / Specific Concern (Optional)", placeholder="e.g., The leaves started turning yellow yesterday.")
    
    st.write("**Voice Input (Optional)**")
    recorded_audio = st.audio_input("Record audio", label_visibility="collapsed")
    uploaded_audio = st.file_uploader("Upload pre-recorded audio", type=["wav", "mp3"])
    
    final_audio_source = recorded_audio if recorded_audio else uploaded_audio
    
    if final_audio_source:
        audio_bytes = final_audio_source.read()
        with open("farm_audio.wav", "wb") as f:
            f.write(audio_bytes)
    else:
        audio_bytes = None

st.divider()

# 7. ANALYSIS EXECUTION & PREMIUM RESULTS
_, center_col, _ = st.columns([1, 2, 1])

with center_col:
    analyze_pressed = st.button("🚀 Run AI Analysis Pipeline", use_container_width=True, type="primary")

if analyze_pressed:
    if uploaded_image is None:
        st.error("System Error: Visual input is required to run the analysis pipeline.")
    else:
        # PREMIUM LOADING ANIMATION
        with st.status("Initializing AI Pipeline...", expanded=True) as status:
            st.write("🔬 Running Convolutional Neural Network on image data...")
            time.sleep(1) # Tiny pause for dramatic UX effect
            disease_result = analyze_leaf(uploaded_image)
            
            st.write("🌐 Connecting to Generative AI reasoning engine...")
            audio_path = "farm_audio.wav" if audio_bytes else None
            final_advice = generate_treatment_plan(disease_result, user_question, selected_lang, audio_path)
            
            status.update(label="Analysis Complete", state="complete", expanded=False)
            st.toast('Treatment plan generated successfully!', icon='✅')
        
        # Save to history
        st.session_state.history.append({
            "time": datetime.now().strftime("%I:%M %p"),
            "disease": disease_result
        })
        
        # PREMIUM RESULTS TABS
        st.subheader("🩺 Diagnostic Report")
        tab1, tab2, tab3 = st.tabs(["🧬 Diagnosis", "📋 Treatment Plan", "🔊 Audio Output"])
        
        with tab1:
            st.metric(label="Primary Detection", value=disease_result, delta="High Confidence")
            
        with tab2:
            st.markdown(final_advice)
            
        with tab3:
            st.write("Generate a localized audio version of the treatment plan.")
            with st.spinner("Synthesizing voice..."):
                try:
                    LANG_CODES = {
                        "English": "en", "Hindi": "hi", "Bengali": "bn", "Telugu": "te",
                        "Marathi": "mr", "Tamil": "ta", "Urdu": "ur", "Gujarati": "gu",
                        "Kannada": "kn", "Malayalam": "ml", "Punjabi": "pa", 
                        "Odia": "or"
                    }
                    lang_code = LANG_CODES.get(selected_lang, "en")
                    clean_text = final_advice.replace('*', '').replace('#', '').replace('_', ' ')
                    tts = gTTS(text=clean_text, lang=lang_code)
                    tts.save("treatment_audio.mp3")
                    st.audio("treatment_audio.mp3", format="audio/mp3")
                except Exception as e:
                    st.warning("Audio synthesis temporarily unavailable.")
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Smart Hospital", page_icon="🏥", layout="wide")

st.markdown("""
<style>
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { display: none; }
.stDeployButton { display: none; }
footer { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1100px !important; }
div[data-testid="stForm"] { border: none; padding: 0; }

div.stButton > button {
    background-color: #1a56db !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    padding: 0.6rem 2rem !important; font-size: 15px !important;
    font-weight: 500 !important; width: 100% !important;
}
div.stButton > button:hover { background-color: #1e429f !important; }

.symptom-chip {
    display: inline-block; padding: 6px 14px; margin: 4px;
    border-radius: 20px; font-size: 13px; font-weight: 500;
    border: 1.5px solid #e5e7eb; background: #f9fafb; color: #374151;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with open('hospital_model.pkl', 'rb') as f:
        return pickle.load(f)

bundle       = load_model()
model        = bundle['model']
scaler       = bundle['scaler']
features     = bundle['features']
cols_to_scale= bundle['cols_to_scale']
dept_map_inv = bundle['dept_map_inv']
gender_map   = bundle['gender_map']
temp_map     = bundle['temp_map']
hr_map       = bundle['hr_map']
dur_map      = bundle['dur_map']
cc_map       = bundle['cc_map']

DEPT_INFO = {
    'Respiratory Medicine': {
        'icon': '🫁', 'color': '#0284c7', 'bg': '#e0f2fe',
        'desc': 'Specialises in conditions affecting the lungs and airways.',
        'next': ['Visit Level 2, Wing B', 'Estimated wait: 15–25 min', 'Please wear a mask']
    },
    'Cardiology': {
        'icon': '❤️', 'color': '#dc2626', 'bg': '#fee2e2',
        'desc': 'Specialises in heart and cardiovascular conditions.',
        'next': ['Visit Level 3, Wing A', 'Estimated wait: 20–30 min', 'Bring any previous ECG reports']
    },
    'Gastroenterology': {
        'icon': '🫃', 'color': '#d97706', 'bg': '#fef3c7',
        'desc': 'Specialises in digestive system and abdominal conditions.',
        'next': ['Visit Level 1, Wing C', 'Estimated wait: 10–20 min', 'Avoid eating before consultation']
    },
    'Neurology': {
        'icon': '🧠', 'color': '#7c3aed', 'bg': '#ede9fe',
        'desc': 'Specialises in brain, spine, and nervous system conditions.',
        'next': ['Visit Level 4, Wing A', 'Estimated wait: 25–35 min', 'Bring list of current medications']
    },
    'General Medicine': {
        'icon': '🩺', 'color': '#059669', 'bg': '#d1fae5',
        'desc': 'Handles general health concerns and non-specialist conditions.',
        'next': ['Visit Level 1, Wing A', 'Estimated wait: 10–15 min', 'Registration desk is open 24/7']
    },
    'Dermatology': {
        'icon': '🔬', 'color': '#b45309', 'bg': '#fef9c3',
        'desc': 'Specialises in skin, hair, and nail conditions.',
        'next': ['Visit Level 2, Wing D', 'Estimated wait: 15–20 min', 'Bring photos of affected area if possible']
    },
}

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_steps = st.columns([1, 2])
with col_logo:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:8px 0;">
        <div style="background:#1a56db;border-radius:12px;padding:10px 14px;font-size:24px;">🏥</div>
        <div>
            <div style="font-size:18px;font-weight:700;color:#111827;">Smart Hospital</div>
            <div style="font-size:12px;color:#6b7280;">AI Patient Guidance</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_steps:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0;padding:12px 0;">
        <div style="text-align:center;flex:1;">
            <div style="width:28px;height:28px;border-radius:50%;background:#1a56db;color:white;font-size:13px;font-weight:600;display:flex;align-items:center;justify-content:center;margin:0 auto;">1</div>
            <div style="font-size:11px;font-weight:600;color:#1a56db;margin-top:4px;">Symptoms</div>
        </div>
        <div style="flex:1;height:2px;background:#e5e7eb;margin-bottom:16px;"></div>
        <div style="text-align:center;flex:1;">
            <div style="width:28px;height:28px;border-radius:50%;background:#e5e7eb;color:#9ca3af;font-size:13px;font-weight:600;display:flex;align-items:center;justify-content:center;margin:0 auto;">2</div>
            <div style="font-size:11px;color:#9ca3af;margin-top:4px;">Details</div>
        </div>
        <div style="flex:1;height:2px;background:#e5e7eb;margin-bottom:16px;"></div>
        <div style="text-align:center;flex:1;">
            <div style="width:28px;height:28px;border-radius:50%;background:#e5e7eb;color:#9ca3af;font-size:13px;font-weight:600;display:flex;align-items:center;justify-content:center;margin:0 auto;">3</div>
            <div style="font-size:11px;color:#9ca3af;margin-top:4px;">Review</div>
        </div>
        <div style="flex:1;height:2px;background:#e5e7eb;margin-bottom:16px;"></div>
        <div style="text-align:center;flex:1;">
            <div style="width:28px;height:28px;border-radius:50%;background:#e5e7eb;color:#9ca3af;font-size:13px;font-weight:600;display:flex;align-items:center;justify-content:center;margin:0 auto;">4</div>
            <div style="font-size:11px;color:#9ca3af;margin-top:4px;">Recommendation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Form ──────────────────────────────────────────────────────────────────────
st.markdown("### Step 1: Tell Us How You Feel")
st.markdown('<p style="color:#6b7280;font-size:14px;">Please fill in the form below so we can guide you to the right department.</p>', unsafe_allow_html=True)

with st.form("triage_form"):
    # Symptoms
    st.markdown("**1. What are your main symptoms?** *(select all that apply)*")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        fever            = st.checkbox("🌡️  Fever")
        cough            = st.checkbox("🤧  Cough")
    with c2:
        headache         = st.checkbox("🤕  Headache")
        chest_pain       = st.checkbox("💔  Chest Pain")
    with c3:
        stomach_pain     = st.checkbox("🤢  Stomach Pain")
        shortness_breath = st.checkbox("😮‍💨  Shortness of Breath")
    with c4:
        nausea_vomiting  = st.checkbox("🤮  Nausea / Vomiting")
        dizziness        = st.checkbox("😵  Dizziness")

    c5, c6 = st.columns([1, 3])
    with c5:
        skin_rash = st.checkbox("🔴  Skin Rash")

    st.markdown("---")

    # Chief complaint + duration
    st.markdown("**2. How long have you had these symptoms?**")
    col_cc, col_dur = st.columns(2)
    with col_cc:
        chief_complaint = st.selectbox("Chief complaint",
            options=list(cc_map.keys()),
            index=0)
    with col_dur:
        duration = st.selectbox("Duration",
            options=list(dur_map.keys()),
            index=1)

    st.markdown("---")

    # Severity
    st.markdown("**3. How would you rate the severity?**")
    col_temp, col_hr = st.columns(2)
    with col_temp:
        temperature_level = st.selectbox("Temperature",
            options=list(temp_map.keys()),
            index=1)
    with col_hr:
        heart_rate_level = st.selectbox("Heart rate",
            options=list(hr_map.keys()),
            index=1)

    st.markdown("---")

    # Medical history
    st.markdown("**4. Do you have any of the following?**")
    ch1, ch2, ch3, ch4 = st.columns(4)
    with ch1: hypertension  = st.checkbox("High Blood Pressure")
    with ch2: heart_disease = st.checkbox("Heart Disease")
    with ch3: asthma        = st.checkbox("Asthma")

    st.markdown("---")

    # Patient info
    st.markdown("**5. Patient information**")
    col_age, col_gen = st.columns(2)
    with col_age:
        age    = st.number_input("Age", min_value=1, max_value=120, value=35)
    with col_gen:
        gender = st.selectbox("Gender", options=['Female', 'Male'])

    submitted = st.form_submit_button("Get AI Recommendation →")

# ── Result ────────────────────────────────────────────────────────────────────
if submitted:
    patient = pd.DataFrame([{
        'age'              : age,
        'gender'           : gender_map.get(gender, 0),
        'fever'            : int(fever),
        'cough'            : int(cough),
        'headache'         : int(headache),
        'chest_pain'       : int(chest_pain),
        'stomach_pain'     : int(stomach_pain),
        'shortness_breath' : int(shortness_breath),
        'nausea_vomiting'  : int(nausea_vomiting),
        'dizziness'        : int(dizziness),
        'skin_rash'        : int(skin_rash),
        'temperature_level': temp_map.get(temperature_level, 1),
        'heart_rate_level' : hr_map.get(heart_rate_level, 1),
        'duration'         : dur_map.get(duration, 1),
        'asthma'           : int(asthma),
        'hypertension'     : int(hypertension),
        'heart_disease'    : int(heart_disease),
        'chief_complaint'  : cc_map.get(chief_complaint, 9)
    }])

    patient_scaled = patient.copy()
    patient_scaled[cols_to_scale] = scaler.transform(patient[cols_to_scale])

    pred  = model.predict(patient_scaled[features])[0]
    proba = model.predict_proba(patient_scaled[features])[0]

    dept_name  = dept_map_inv[pred]
    confidence = proba[pred] * 100
    info       = DEPT_INFO[dept_name]

    st.markdown("---")
    st.markdown("### AI Recommendation")
    st.markdown('<p style="color:#6b7280;font-size:13px;">Based on the information you provided</p>', unsafe_allow_html=True)

    res_col, prob_col = st.columns([3, 2])

    with res_col:
        st.markdown(f"""
        <div style="background:{info['bg']};border-radius:16px;padding:24px 28px;border:1px solid {info['color']}33;">
            <div style="font-size:32px;margin-bottom:8px;">{info['icon']}</div>
            <div style="font-size:22px;font-weight:700;color:{info['color']};margin-bottom:6px;">{dept_name}</div>
            <div style="font-size:14px;color:#374151;margin-bottom:16px;">Our AI suggests you visit the {dept_name} Department.</div>
            <div style="font-size:12px;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">Why?</div>
            <div style="font-size:13px;color:#4b5563;margin-bottom:16px;">{info['desc']} Your reported symptoms and vitals match patients typically directed to this department.</div>
            <div style="font-size:12px;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">What to do next?</div>
            {''.join(f'<div style="font-size:13px;color:#374151;margin-bottom:4px;">📍 {step}</div>' for step in info['next'])}
            <div style="margin-top:16px;padding:10px 14px;background:rgba(0,0,0,0.04);border-radius:8px;font-size:11px;color:#6b7280;">
                This is an AI suggestion, not a medical diagnosis. Please consult a doctor for further evaluation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with prob_col:
        st.markdown("**Confidence by department:**")
        sorted_depts = sorted(dept_map_inv.items(), key=lambda x: proba[x[0]], reverse=True)
        for idx, dname in sorted_depts:
            pct      = proba[idx] * 100
            dinfo    = DEPT_INFO[dname]
            is_top   = dname == dept_name
            bar_color = dinfo['color'] if is_top else '#d1d5db'
            fw        = '700' if is_top else '400'
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;font-size:13px;font-weight:{fw};color:{'#111827' if is_top else '#6b7280'};">
                    <span>{dinfo['icon']} {dname}</span>
                    <span>{pct:.1f}%</span>
                </div>
                <div style="background:#f3f4f6;border-radius:4px;height:6px;margin-top:4px;">
                    <div style="background:{bar_color};height:100%;border-radius:4px;width:{pct}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:16px;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:12px 16px;font-size:12px;color:#166534;">
            <strong>Model info:</strong> KNN (k=7) · 102,000 patients · 98% test accuracy · Future Classroom ML
        </div>
        """, unsafe_allow_html=True)

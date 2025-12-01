import json
import numpy as np
import streamlit as st

from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import joblib


# PATHS
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"


# NODE 1 – TEXT MODEL (Glove LSTM pipeline)
@st.cache_resource
def load_node1_models():
    """
    Load text classification model + tokenizer + config.
    Files expected in models/:
      - suicide_config.json
      - suicide_tokenizer.json
      - glove_bilstm_suicide_model.keras
    """
    with open(MODEL_DIR / "suicide_config.json", "r") as f:
        config = json.load(f)

    max_len = config.get("MAX_LEN", 200)

    with open(MODEL_DIR / "suicide_tokenizer.json", "r") as f:
        tok_json = f.read()
    tokenizer = tokenizer_from_json(tok_json)

    text_model = load_model(MODEL_DIR / "glove_bilstm_suicide_model.keras")
    return text_model, tokenizer, max_len


def node1_predict(text_model, tokenizer, max_len, text: str) -> float:
    """
    Run the Node 1 model on a single text string.
    Returns probability of the "higher-risk" class (as trained).
    """
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    prob = float(text_model.predict(pad, verbose=0)[0][0])
    return prob


def node1_zone(prob: float):
    """
    Convert probability into a friendly zone + message.
    """
    if prob < 0.20:
        zone = "Green Zone 💚"
        msg = (
            "Your text looks closer to calmer patterns in our research dataset.\n\n"
            "Hurrah, you’re doing great – keep it up! 🌱"
        )
    elif prob < 0.50:
        zone = "Yellow Zone 💛"
        msg = (
            "There might be some emotional tension.\n\n"
            "It might help to talk with someone you trust, take a break, "
            "or do something relaxing."
        )
    else:
        zone = "Red Zone ❤️‍🩹"
        msg = (
            "**Your text looks similar to messages written during very tough moments.**\n\n"
            "You deserve support. Please consider talking to someone you trust, contacting your GP, "
            "or calling **111** for urgent medical advice.\n\n"
            "If you feel in immediate danger or unable to stay safe, call **999** right away."
        )
    return zone, msg


# NODE 2 – CLINICAL MODEL (LightGBM pipeline)
@st.cache_resource
def load_node2_assets():
    """
    Load the clinical LightGBM model + frequency maps + config.
    Files expected in models/:
      - clinical_lgbm_treatment_deploy.pkl
      - clinical_country_freq_map.pkl
      - clinical_occupation_freq_map.pkl
      - clinical_lgbm_deploy_config.json
    """
    model = joblib.load(MODEL_DIR / "clinical_lgbm_treatment_deploy.pkl")
    country_freq_map = joblib.load(MODEL_DIR / "clinical_country_freq_map.pkl")
    occupation_freq_map = joblib.load(MODEL_DIR / "clinical_occupation_freq_map.pkl")

    with open(MODEL_DIR / "clinical_lgbm_deploy_config.json", "r") as f:
        cfg = json.load(f)

    feature_order = cfg["feature_order"]
    threshold = cfg.get("threshold", 0.5)

    # For dropdowns (assumes index of series is the category labels)
    country_classes = sorted(list(country_freq_map.index))
    occupation_classes = sorted(list(occupation_freq_map.index))

    return (
        model,
        feature_order,
        threshold,
        country_freq_map,
        occupation_freq_map,
        country_classes,
        occupation_classes,
    )


def encode_node2_from_inputs(
    answers: dict,
    feature_order,
    country_freq_map,
    occupation_freq_map,
) -> np.ndarray:
    """
    Build a 1D feature vector in the exact order given by feature_order.
    answers contains raw fields including:
      - 'country_str'
      - 'occupation_str'
      - 'gender_binary'
      - 'familyhistory'
      - 'careoptions'
      - 'mentalhealthinterview'
      - 'selfemployed'
    """
    vec = []

    # Lowercase strings to match how freq maps were built
    country_str = answers.get("country_str", "").strip().lower()
    occ_str = answers.get("occupation_str", "").strip().lower()

    cf = float(country_freq_map.get(country_str, 0.0))
    of = float(occupation_freq_map.get(occ_str, 0.0))

    for feat in feature_order:
        if feat == "country_freq":
            vec.append(cf)
        elif feat == "occupation_freq":
            vec.append(of)
        else:
            vec.append(float(answers.get(feat, 0.0)))

    return np.array(vec, dtype=float).reshape(1, -1)


def node2_message(prob: float, threshold: float):
    """
    Interpret the clinical model probability with soft, non-diagnostic language.
    """
    high_cut = min(1.0, threshold + 0.10)
    mid_low = max(0.0, threshold - 0.10)

    if prob >= high_cut:
        title = "High attention recommended 🔴"
        msg = (
            "This pattern looks similar to people who, in our dataset, were receiving mental health treatment.\n\n"
            "This is **not a diagnosis**, but it would be wise to seek support: contacting a GP, counsellor, "
            "or mental health service could really help."
        )
    elif prob >= mid_low:
        title = "Some signs of difficulty 💛"
        msg = (
            "The probability is around the range where people in our dataset sometimes did receive treatment.\n\n"
            "It may be worth keeping a closer eye on things, talking with someone you trust, and considering "
            "support if difficulties continue."
        )
    else:
        title = "Lower indication for treatment 🌱"
        msg = (
            "This pattern does **not** strongly match the treatment group in our dataset.\n\n"
            "However, the model is limited and cannot see the full picture. If someone is struggling, it's always "
            "okay to seek help, regardless of what the model says."
        )

    return title, msg


# STREAMLIT UI
st.set_page_config(
    page_title="StressGuardian",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 StressGuardian")
st.write(
    "A gentle, research-based tool that uses AI to explore patterns related to stress and emotional wellbeing. "
    "This is a **prototype**, not a medical device."
)

st.warning(
    "⚠️ If you ever feel unsafe or in immediate danger, call **999**. "
    "For urgent medical advice in the UK, call **111**."
)

tab1, tab2, tab3 = st.tabs(
    [
        "📝 Text Check-in (Node 1)",
        "📋 Clinical Support Suggestion (Node 2)",
        "⌚ Wearable Stress (Node 3 – Future Work)",
    ]
)


# TAB 1: Node 1 – Text
with tab1:
    st.header("📝 Emotional Text Check-in")

    txt = st.text_area(
        "Write about how you're feeling:",
        placeholder="Example: I’ve been feeling really overwhelmed lately and I don't know what to do...",
        height=180,
    )

    if st.button("Check my text"):
        if not txt.strip():
            st.error("Please type something first so I can reflect on it with you 💬")
        else:
            with st.spinner("Thinking about your text with care..."):
                text_model, tokenizer, max_len = load_node1_models()
                prob = node1_predict(text_model, tokenizer, max_len, txt)

            zone, msg = node1_zone(prob)

            st.subheader("Your current zone")
            st.markdown(f"### {zone}")
            st.write(msg)
            st.caption(f"(Model probability for higher-risk pattern: {prob:.3f})")


# TAB 2: Node 2 – Clinical (LightGBM)
with tab2:
    st.header("📋 Clinical Support Suggestion (Node 2)")

    (
        clinical_model,
        feature_order,
        clinical_threshold,
        country_freq_map,
        occupation_freq_map,
        country_classes,
        occupation_classes,
    ) = load_node2_assets()

    st.write(
        "Answer these questions. The clinical model will estimate how similar this pattern is to people "
        "who were receiving mental health treatment in the dataset."
    )

    with st.form("node2_form"):

        st.markdown("### 1️⃣ Background")

        gender = st.selectbox(
            "Gender",
            options=["male", "female"],
            help="For the model we convert this to a binary feature.",
        )
        # in training: gender_binary = 1 if male else 0
        gender_binary = 1 if gender.lower() == "male" else 0

        selfemployed = st.selectbox(
            "Are you self-employed?",
            options=["no", "yes"],
        )

        # 🌡️ 3-LEVEL STRESS QUESTION
        st.markdown("### 🌡️ Stress Level")
        stress_question = st.selectbox(
            "How stressed have you felt in the last week?",
            [
                "No stress",
                "Moderate stress",
                "Extreme stress",
            ]
        )
        stress_map = {
            "No stress": 0,
            "Moderate stress": 1,
            "Extreme stress": 2,
        }
        stress_level = stress_map[stress_question]

        st.markdown("### 2️⃣ History & Access to Care")

        familyhistory = st.selectbox(
            "Family history of mental health issues?",
            options=["no", "yes"],
        )

        mentalhealthinterview = st.selectbox(
            "Would you feel comfortable discussing mental health at work?",
            options=["no", "yes"],
        )

        careoptions = st.selectbox(
            "Do you feel you have access to mental health care options?",
            options=["no", "yes"],
        )

        st.markdown("### 3️⃣ Country & Occupation")

        country_choice = st.selectbox(
            "Country (from dataset labels)",
            options=country_classes,
        )

        occupation_choice = st.selectbox(
            "Occupation (from dataset labels)",
            options=occupation_classes,
        )

        submitted = st.form_submit_button("Analyze")

    if submitted:
        # yes/no -> 1/0 as in training
        yn_map = {"yes": 1, "no": 0}

        answers = {
            "gender_binary": gender_binary,
            "selfemployed": yn_map[selfemployed],
            "familyhistory": yn_map[familyhistory],
            "careoptions": yn_map[careoptions],
            "mentalhealthinterview": yn_map[mentalhealthinterview],
            "country_str": country_choice,
            "occupation_str": occupation_choice,
        }

        X_vec = encode_node2_from_inputs(
            answers,
            feature_order,
            country_freq_map,
            occupation_freq_map,
        )

        with st.spinner("Analyzing support needs using the clinical model..."):
            prob = float(clinical_model.predict_proba(X_vec)[0][1])

        title, msg = node2_message(prob, clinical_threshold)

        # 🌡️ Stress-based message adjustment
        if stress_level == 2:  # Extreme stress
            msg += (
                "\n\n💥 **You reported feeling extremely stressed.** "
                "Even if the model probability is low, extreme stress can take a real toll. "
                "It may help to talk with someone supportive, take a break, or reach out to your GP or a care service."
            )
        elif stress_level == 1:  # Moderate stress
            msg += (
                "\n\n🌤️ You also mentioned *moderate stress*. "
                "Practicing self-care, rest, or brief relaxation during the week could be helpful."
            )
        # if stress_level == 0 → no extra text

        st.subheader("🧾 Result")
        st.markdown(f"### {title}")
        st.write(msg)
        st.info(f"**Model probability (treatment-like pattern)**: {prob:.3f}")


# TAB 3: Node 3 – Wearable (Future Work)
with tab3:
    st.header("⌚ Wearable Stress (Node 3 – Future Work)")

    st.write(
        """
        In the future, **StressGuardian** will include a wearable-based stress module that connects directly
        to devices like:
        - Fitbit  
        - Apple Watch  
        - Garmin  
        - Google Fit  

        Instead of asking users to download CSV files, the app will use **secure APIs** (OAuth) to read:
        - Heart rate (HR)  
        - Movement / accelerometer data  
        - Electrodermal activity (EDA), if available  
        - Skin temperature  
        """
    )

    st.markdown("### Why use this instead of built-in smartwatch stress features?")

    st.write(
        """
        While many smartwatches already offer built-in stress scores, they come with **several limitations**:

        1. **Black-box models**  
           - Commercial devices don't explain how their stress scores are calculated.  
           - You can't see which signals contributed to the result or how to adjust it.

        2. **No mental health integration**  
           - Built-in stress scores are separate from emotional check-ins or clinical support.  
           - They don't connect to mood, surveys, or patterns over time.

        3. **Not tuned for high-stress roles**  
           - Most devices are optimized for general wellness, not shift workers, healthcare staff, or academic stress.  
           - StressGuardian is based on real wearable data from nurses.

        4. **No access to the data behind the score**  
           - You can't export the model probability or see a timeline of stress.  
           - You just get a daily summary or vague label.

        5. **No user control or context**  
           - Built-in apps don’t let users add context (e.g. “this was during a shift” or “after caffeine”).  
           - StressGuardian can be extended to allow user journaling, event tags, or sensitivity adjustment.

        📊 **In short**: StressGuardian offers **research-level transparency**, integration with emotional and clinical models, and the flexibility to grow into a tool for **personal reflection, study, or clinical trials**.
        """
    )

    st.markdown("### Planned processing pipeline")

    st.markdown(
        """
        1. **Connect smartwatch via API**  
           - User taps “Connect my device” and approves access.  
           - StressGuardian receives a token to read recent sensor data securely.

        2. **Collect raw sensor streams**  
           - HR, accelerometer (X, Y, Z), EDA, temperature, etc.  
           - Data is pulled for a recent time window (e.g. last 30–60 minutes).

        3. **Windowing**  
           - The signals are split into short windows (e.g. 10–30 seconds).  
           - Each window is treated as one sample for stress prediction.

        4. **Two parallel feature paths**  
           - **Tabular path (for XGBoost / tree models)**:  
             - Summary statistics per window (mean, std, min, max, range, slopes, spike counts, HRV-like metrics).  
           - **Sequence path (for RNN / Transformer)**:  
             - The full normalised time series for each window (shape `[time_steps, channels]`).

        5. **Model inference**  
           - **Tabular model** → probability of “high physiological stress”.  
           - **Sequence model** → probability based on temporal patterns.

        6. **Ensemble**  
           - Combine both outputs, for example:  
             - `p_final = 0.8 * p_tabular + 0.2 * p_sequence`  
           - Apply a threshold to classify each window as **higher** or **lower** stress.

        7. **Feedback to user**  
           - Plot stress probability over time as a line chart.  
           - Provide gentle wording such as:  
             - “Your body looked calmer in these periods.”  
             - “Here your body showed patterns closer to high stress.”  
           - Offer suggestions like breaks, breathing exercises, or reaching out for support.
        """
    )

    st.markdown("### Why it's marked as future work")

    st.write(
        """
        - The current research models for Node 3 were trained on a **nurse wearable dataset** and saved with an
          older Keras version that uses `Lambda` layers.  
        - Newer Keras versions (used in this app) have compatibility and safety restrictions that make loading
          that exact RNN model unreliable in this environment.  
        - To keep this prototype **stable and safe**, the wearable module is presented here as an **architecture and design**
          rather than an active prediction feature.
        """
    )

    st.success(
        "Hurrah – Node 1 and Node 2 are already working end-to-end, and Node 3 has a clear future roadmap. 🚀"
    )

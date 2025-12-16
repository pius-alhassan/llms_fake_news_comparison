import streamlit as st
import requests
from time import sleep

API_BASE = "http://localhost:5000"

# ------------------ PAGE CONFIG ------------------

st.set_page_config(
    page_title="Multimodal Fake News Detector",
    layout="wide"
)

st.title("🧠 Multimodal Fake News Detection System")
st.markdown(
    "This system demonstrates the ability of different Large Language Models (LLMs) "
    "to detect fake news from **video data**."
)

# ------------------ SESSION STATE INIT ------------------
#------------------------------------------------
# Video Mode Session State Initialization
# -----------------------------------------------
if "videos" not in st.session_state:
    st.session_state.videos = None

if "prediction_started" not in st.session_state:
    st.session_state.prediction_started = False

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "selected_video_id" not in st.session_state:
    st.session_state.selected_video_id = None
#------------------------------------------------
# Text Mode Session State Initialization
# -----------------------------------------------
if "texts" not in st.session_state:
    st.session_state.texts = None

if "selected_text_id" not in st.session_state:
    st.session_state.selected_text_id = None

if "text_prediction_started" not in st.session_state:
    st.session_state.text_prediction_started = False

if "text_prediction_result" not in st.session_state:
    st.session_state.text_prediction_result = None

if "prev_text_id" not in st.session_state:
    st.session_state.prev_text_id = None

# ------------------ SIDEBAR ------------------

st.sidebar.header("Configuration")

model = st.sidebar.selectbox(
    "Select LLM",
    options=["gemini", "gpt4o", "mistral"]
)

batch_size = st.sidebar.slider(
    "Number of videos to sample",
    min_value=5,
    max_value=20,
    value=10
)

if "selected_model" not in st.session_state:
    st.session_state.selected_model = model

if "prev_text_model" not in st.session_state:
    st.session_state.prev_text_model = model

# Detect model change → reset prediction
if st.session_state.selected_model != model:
    st.session_state.selected_model = model
    st.session_state.prediction_started = False
    st.session_state.prediction_result = None

if st.sidebar.button("Load Random Videos"):
    try:
        response = requests.get(
            f"{API_BASE}/api/videos/random",
            params={"batch": batch_size},
            timeout=15
        )
        response.raise_for_status()
        st.session_state.videos = response.json()["videos"]

        # Reset prediction state
        st.session_state.prediction_started = False
        st.session_state.prediction_result = None
        st.session_state.selected_video_id = None

    except requests.exceptions.Timeout:
        st.error("⏱️ Server timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        st.error("🌐 Cannot connect to backend server.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

# ------------------ MAIN PANEL ------------------
mode = st.tabs(["🎥 Video Mode", "📰 Text Mode"])
with mode[0]:

    if st.session_state.videos:

        st.subheader("🎞️ Sampled Videos")

        video_map = {
            f'#{v["id"]} | {v["file_name"]}': v
            for v in st.session_state.videos
        }

        selection = st.selectbox(
            "Select a video to analyze",
            options=list(video_map.keys())
        )

        selected_video = video_map[selection]

        # Detect video change → reset prediction
        if st.session_state.selected_video_id != selected_video["id"]:
            st.session_state.selected_video_id = selected_video["id"]
            st.session_state.prediction_started = False
            st.session_state.prediction_result = None

        # ------------------ LAYOUT ------------------

        video_col, pred_col = st.columns([1, 2])

        # ------------------ VIDEO COLUMN ------------------

        with video_col:
            st.markdown("### 🎥 Video Preview")

            video_path = f"data/raw/video/faceforensics/{selected_video['file_name']}"
            st.video(video_path)

            st.markdown("### 🧑 Take a guess")
            user_guess = st.radio(
                "Do you think this video is real or fake?",
                ["REAL", "FAKE"],
                horizontal=True,
                key="user_guess"
            )

            st.markdown("### 📄 Video Metadata")
            st.write("**Resolution:**", f'{selected_video["width"]}×{selected_video["height"]}')
            st.write("**Codec:**", selected_video["codec"])

        # ------------------ PREDICTION COLUMN ------------------

        with pred_col:

            # ---- BEFORE PREDICTION ----
            if not st.session_state.prediction_started:

                st.markdown("<br><br><br>", unsafe_allow_html=True)

                center = st.columns([1, 2, 1])[1]
                with center:
                    if st.button("🔍 Run Prediction", use_container_width=True):
                        st.session_state.prediction_started = True
                        st.session_state.prediction_result = None
                        st.rerun()

            # ---- RUN PREDICTION ----
            if st.session_state.prediction_started and st.session_state.prediction_result is None:

                with st.spinner("Analyzing video frames..."):
                    try:
                        pred = requests.get(
                            f"{API_BASE}/api/videos/predict/{selected_video['id']}",
                            params={"model": model},
                            timeout=60
                        )
                        pred.raise_for_status()
                        st.session_state.prediction_result = pred.json()

                    except requests.exceptions.Timeout:
                        st.error("⏱️ Prediction timed out.")
                        st.session_state.prediction_started = False
                        st.stop()

                    except requests.exceptions.ConnectionError:
                        st.error("🌐 Network error.")
                        st.session_state.prediction_started = False
                        st.stop()

                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        st.session_state.prediction_started = False
                        st.stop()

                st.rerun()

            # ---- DISPLAY RESULTS ----
            if st.session_state.prediction_result:

                result = st.session_state.prediction_result

                st.markdown("## 🔍 Final Results")

                st.markdown(f"**🧪 Ground Truth:** {result['ground_truth']}")
                st.markdown(f"**🧑 Your Guess:** {st.session_state['user_guess']}")
                st.markdown(f"**🤖 LLM Prediction:** {result['prediction']}")

                st.progress(result["confidence"] / 100)
                st.caption(f"LLM Confidence: {result['confidence']}%")

                st.markdown("### 🧠 LLM Explanation")
                st.info(result["explanation"])

                st.markdown("### ✅ Accuracy Check")
                st.write(
                    "👤 You:",
                    "Correct" if st.session_state["user_guess"] == result["ground_truth"] else "Incorrect"
                )
                st.write(
                    "🤖 LLM:",
                    "Correct" if result["prediction"].upper() == result["ground_truth"] else "Incorrect"
                )

    else:
        st.info("Load random videos using the sidebar to begin.")

# ------------------ TEXT MODE -------------------
with mode[1]:

    st.subheader("📰 Text Fake News Detection (LIAR Dataset)")

# ---------------- Load Statements ----------------
col_cfg, col_main = st.columns([1, 3])

with col_cfg:
    if st.button("📄 Load Random Statements"):
        try:
            response = requests.get(
                f"{API_BASE}/api/text/random",
                params={"batch": 10, "split": "test"},
                timeout=15
            )
            response.raise_for_status()

            st.session_state.texts = response.json()["texts"]

            # Reset ALL text-related state
            st.session_state.selected_text_id = None
            st.session_state.text_prediction_started = False
            st.session_state.text_prediction_result = None
            st.session_state.prev_text_id = None
            st.session_state.prev_text_model = None

            st.rerun()

        except Exception as e:
            st.error(f"Failed to load texts: {e}")

# ---------------- Main Panel ----------------
if st.session_state.texts:

    text_map = {
        f'#{t["id"]} | {t["speaker"]}': t
        for t in st.session_state.texts
    }

    selection = st.selectbox(
        "Select a statement",
        options=list(text_map.keys())
    )

    selected_text = text_map[selection]

    # ---------------- Context Change Detection ----------------
    if (
        st.session_state.prev_text_model != model or
        st.session_state.prev_text_id != selected_text["id"]
    ):
        st.session_state.text_prediction_started = False
        st.session_state.text_prediction_result = None

    st.session_state.prev_text_model = model
    st.session_state.prev_text_id = selected_text["id"]

    # ---------------- Display Statement ----------------
    st.markdown("### 📝 Statement")
    st.info(selected_text["text"])

    st.markdown(f"**Speaker:** {selected_text['speaker']}")
    st.markdown(f"**Context:** {selected_text['context']}")

    # ---------------- User Guess ----------------
    user_guess = st.radio(
        "Do you think this statement is real or fake?",
        ["REAL", "FAKE"],
        horizontal=True,
        key="text_user_guess"
    )

    # ---------------- Prediction Area ----------------
    st.markdown("---")

    if not st.session_state.text_prediction_started:
        center = st.columns([1, 2, 1])[1]
        with center:
            if st.button("🔍 Run Text Prediction", use_container_width=True):
                st.session_state.text_prediction_started = True
                st.session_state.text_prediction_result = None
                st.rerun()

    # ---------------- Execute Prediction ----------------
    if (
        st.session_state.text_prediction_started and
        st.session_state.text_prediction_result is None
    ):
        with st.spinner("Analyzing text..."):
            try:
                pred = requests.post(
                    f"{API_BASE}/api/text/predict",
                    json={
                        "text": selected_text["text"],
                        "model": model,
                        "ground_truth": selected_text["label"]
                    },
                    timeout=30
                )
                pred.raise_for_status()
                st.session_state.text_prediction_result = pred.json()

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.session_state.text_prediction_started = False
                st.stop()

        st.rerun()

    # ---------------- Show Results ----------------
    if st.session_state.text_prediction_result:
        result = st.session_state.text_prediction_result

        st.markdown("## 🔍 Final Results")

        st.markdown(f"**🧪 Ground Truth:** {selected_text['label']}")
        st.markdown(f"**🧑 Your Guess:** {st.session_state.text_user_guess}")
        st.markdown(f"**🤖 LLM Prediction:** {result['prediction']}")

        st.progress(result["confidence"] / 100)
        st.caption(f"LLM Confidence: {result['confidence']}%")

        st.markdown("### 🧠 LLM Explanation")
        st.info(result["explanation"])

        st.markdown("### ✅ Accuracy Check")
        st.write(
            "👤 You:",
            "Correct" if st.session_state.text_user_guess == selected_text["label"] else "Incorrect"
        )
        st.write(
            "🤖 LLM:",
            "Correct" if result["prediction"].upper() == selected_text["label"] else "Incorrect"
        )

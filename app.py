import streamlit as st
import cv2
import time
import threading
import json
from collections import deque
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
import re


CLIP_FPS = 2
CLIP_LENGTH = 8

# ==== Default Activity Definitions ====
DEFAULT_ACTIVITIES = [
    {
        "key": "fall_detected",
        "label": "Fall",
        "description": "The person suddenly collapses or ends up on the floor or ground in an unnatural position."
    },
    {
        "key": "attempt_to_get_up",
        "label": "Attempt to Get Up",
        "description": "The person is on the floor and is actively trying to stand up, push themselves up, or use nearby objects to lift themselves."
    },
    {
        "key": "eating_non_food",
        "label": "Eating Non-Food",
        "description": "The person places or attempts to place a non-edible object (e.g. remote, paper, fabric) into their mouth."
    }
]

# ==== Prompt Generation Functions ====
def generate_monitoring_prompt(activities):
    prompt_lines = [
        "You are analyzing a video from a camera monitoring an elderly person.",
        "Your task is to detect whether any of the following events have occurred:"
    ]
    json_structure = "{\n"
    for i, activity in enumerate(activities):
        prompt_lines.append(f'\n{i+1}. {activity["label"]}: {activity["description"]}')
        json_structure += f'  "{activity["key"]}": true | false,\n'
        json_structure += f'  "{activity["key"]}_reason": "Short explanation or null",\n'
    json_structure = json_structure.rstrip(',\n') + "\n}"
    prompt_lines.append("\nFor the given video, reply only with the following JSON format:")
    prompt_lines.append(json_structure)
    return "\n".join(prompt_lines)


REFINEMENT_PROMPT = """You are an assistant for a smart home system designed to monitor the well-being of elderly individuals. Your primary function is to configure new activities to be monitored.

A user will provide a brief phrase or sentence describing an activity they want to track. Your task is to process this input and generate a single, valid JSON object representing that activity for the system's backend.

The JSON object must follow this precise structure:

"key": A machine-readable identifier in snake_case. It must be short, descriptive, and contain only lowercase letters and underscores. (e.g., medication_taken, fall_detected).
"label": A human-readable, Title Case string for display in user interfaces. This should be a clean and concise name for the activity. (e.g., Medication Taken, Fall Detected).
"description": A short `description` in a single sentnce of the new activity. (e.g., "The person suddenly collapses on the floor or ground in an unnatural position", "The person places or attempts to place a non-edible object").

Reply with a single JSON object and nothing else.

Activity described by the user: "{user_input}"
"""

# ==== Session State Initialization ====
if 'activities' not in st.session_state:
    st.session_state.activities = DEFAULT_ACTIVITIES.copy()
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
    st.session_state.stop_event = threading.Event()
    st.session_state.threads = []
    st.session_state.frame_buffer = deque(maxlen=CLIP_LENGTH)
    st.session_state.detection_logs = deque(maxlen=5)
    st.session_state.latest_frame_container = [None]
    # **NEW**: Store the parsed JSON from the latest valid detection
    st.session_state.last_detection_data: dict = {}
    st.session_state.last_processed_log = None


# ==== Model Load ====
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    model_path = "mlx-community/gemma-3n-E4B-it-4bit"
    model, processor = load(model_path)
    return model, processor, model.config

model, processor, config = load_model()

# ==== Threading Functions ====
def capture_frames(source, stop_event, frame_buffer, latest_frame_container):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.toast("Error: Could not open video source.", icon="🚨")
        return
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            continue
        frame_buffer.append(frame.copy())
        latest_frame_container[0] = frame.copy()
        time.sleep(1 / CLIP_FPS)
    cap.release()

def run_inference(stop_event, frame_buffer, detection_logs, model, processor, config, prompt):
    while not stop_event.is_set():
        if len(frame_buffer) >= CLIP_LENGTH:
            image_list = list(frame_buffer)
            messages = apply_chat_template(processor, config, prompt, num_images=len(image_list))
            try:
                output = generate(model, processor, messages, image=image_list, verbose=False)
                print(output.text, time.time())
                # The deque append is thread-safe
                detection_logs.appendleft(output.text)
            except Exception as e:
                detection_logs.appendleft(f'{{"error": "Error during inference: {e}"}}')


# **NEW**: Helper function to display the status list
def display_activity_status():
    st.subheader("Activity Status")
    
    if not st.session_state.activities:
        st.caption("No activities defined.")
        return

    for activity in st.session_state.activities:
        key = activity['key']
        label = activity['label']
        
        # Check the latest detection data for this activity's key
        if st.session_state.last_detection_data.get(key) is True:
            reason = st.session_state.last_detection_data.get(f"{key}_reason", "No reason provided.")
            st.success(f"✅ {label}: Detected", icon="✅")
            with st.container(border=True):
                 st.caption(f"Reason: {reason}")
        else:
            st.info(f"⚪ {label}: Monitoring...", icon="⚪️")


# ==== GUI ====
st.set_page_config(layout="wide")
st.title("Elderly Monitoring Dashboard")

main_col1, main_col2 = st.columns(2)

with main_col1:
    st.header("🎬 Live Feed & Controls")
    video_placeholder = st.empty()
    
    # **NEW**: Placeholder for the dynamic status list
    status_placeholder = st.empty()

    source_options = {"Webcam (default)": 0, "USB Camera": 1}
    source_label = st.selectbox("Select Video Source", list(source_options.keys()), disabled=st.session_state.is_running)
    source_value = source_options[source_label]

    btn_col1, btn_col2 = st.columns(2)
    if btn_col1.button("▶️ Start Monitoring", disabled=st.session_state.is_running or not st.session_state.activities, use_container_width=True):
        st.session_state.is_running = True
        st.session_state.stop_event.clear()
        st.session_state.detection_logs.clear()
        st.session_state.frame_buffer.clear()
        st.session_state.latest_frame_container[0] = None
        # **NEW**: Reset detection data on start
        st.session_state.last_detection_data = {}
        st.session_state.last_processed_log = None

        current_monitoring_prompt = generate_monitoring_prompt(st.session_state.activities)
        t1 = threading.Thread(target=capture_frames, args=(source_value, st.session_state.stop_event, st.session_state.frame_buffer, st.session_state.latest_frame_container), daemon=True)
        t2 = threading.Thread(target=run_inference, args=(st.session_state.stop_event, st.session_state.frame_buffer, st.session_state.detection_logs, model, processor, config, current_monitoring_prompt), daemon=True)
        st.session_state.threads = [t1, t2]
        for t in st.session_state.threads:
            t.start()
        st.rerun()

    if btn_col2.button("⏹️ Stop Monitoring", disabled=not st.session_state.is_running, use_container_width=True):
        st.session_state.stop_event.set()

    with st.expander("⚙️ Manage Monitored Activities"):
        # (This section is unchanged)
        activities_to_remove = []
        if not st.session_state.activities:
            st.caption("No activities defined.")
        for activity in st.session_state.activities:
            act_col1, act_col2 = st.columns([4, 1])
            act_col1.markdown(f"**{activity['label']}**: _{activity['description']}_")
            if act_col2.button("🗑️", key=f"remove_{activity['key']}", use_container_width=True):
                activities_to_remove.append(activity)
        if activities_to_remove:
            st.session_state.activities = [act for act in st.session_state.activities if act not in activities_to_remove]
            st.rerun()
        st.subheader("Add a New Activity")
        new_activity_desc = st.text_area("Describe the activity:", placeholder="e.g., 'the person is drinking water'")
        if st.button("✨ Refine and Add Activity", use_container_width=True):
            if new_activity_desc:
                with st.spinner("Asking AI to refine the activity..."):
                    try:
                        refinement_prompt = REFINEMENT_PROMPT.format(user_input=new_activity_desc)
                        messages = apply_chat_template(processor, config, refinement_prompt)
                        output = generate(model, processor, messages)
                        print(output.text)
                        
                        # Robustly find the JSON block
                        match = re.search(r"\{.*\}", output.text, re.DOTALL)
                        if not match:
                            # Fallback for ```json ... ``` format
                            match = re.search(r"```json\s*(\{.*\})\s*```", output.text, re.DOTALL)
                            if match:
                                json_str = match.group(1)
                            else:
                                raise ValueError("No JSON object found in the response.")
                        else:
                            json_str = match.group(0)

                        refined_activity = json.loads(json_str)
                        
                        if "key" in refined_activity and "label" in refined_activity and "description" in refined_activity:
                            st.session_state.activities.append(refined_activity)
                            st.success(f"Added new activity: {refined_activity['label']}")
                            st.rerun()
                        else:
                            st.error("The AI's response was not in the correct format. Please try again.")
                    except (json.JSONDecodeError, ValueError) as e:
                        st.error(f"Could not parse the AI's response: {e}. Please try rephrasing.")
                        st.code(output.text, language="text")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
        else:
            st.warning("Please describe the activity first.")

with main_col2:
    st.header("📋 Detection Logs")
    log_placeholder = st.empty()

# ==== Main Application Loop ====
if not st.session_state.is_running:
    video_placeholder.info("Video feed will appear here when monitoring starts.", icon="📹")
    with status_placeholder.container():
        display_activity_status()

while st.session_state.is_running:
    # Display video
    frame = st.session_state.latest_frame_container[0]
    if frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    # **NEW**: Check for and process new detections
    if st.session_state.detection_logs and st.session_state.detection_logs[0] != st.session_state.last_processed_log:
        latest_log = st.session_state.detection_logs[0]
        st.session_state.last_processed_log = latest_log
        try:
            match = re.search(r"\{.*\}", latest_log, re.DOTALL)
            if match:
                json_str = match.group(0)
                st.session_state.last_detection_data = json.loads(json_str)
        except json.JSONDecodeError:
            # The log was not valid JSON, so we don't update the status
            pass
    
    # Update the UI placeholders
    with status_placeholder.container():
        display_activity_status()

    with log_placeholder.container():
        st.markdown("---")
        if not st.session_state.detection_logs:
            st.caption("Waiting for first detection...")
        for log in list(st.session_state.detection_logs):
            st.code(log, language="json")

    # Check for stop signal
    if st.session_state.stop_event.is_set():
        st.session_state.is_running = False
        st.rerun()

    time.sleep(0.1)
# gemmeye



This document details the design and implementation of a real-time elder care monitoring application powered by a local Vision-Language Model (VLM). The system captures a video stream from a selected camera, processes the video frames to detect specific high-risk behaviors using a VLM, and displays the results on a local dashboard interface. The primary goal is to assist in remote elder monitoring while preserving privacy by keeping all computation on the edge.

## Features

* Detect key events such as falls, attempts to get up, and eating non-food items.

* Maintain low latency for real-time responsiveness.

* Enable video source selection.

* Provide a local dashboard for visualization and control.

* Ensure all processing is done locally for privacy.

## Design

The system is composed of several interrelated components that work together to enable real-time detection. A video source (a webcam for this demo) provides the raw input. Frames from this source are continuously captured and stored in a circular buffer, implemented using Python’s collections.deque, which holds a fixed-length sliding window of the most recent frames. The inference engine monitors this buffer and, once it contains enough frames, builds a list of frames and formats it into a prompt for Gemma. The model is deployed locally using the mlx_vlm library with the gemma-3n-E2B-it-4bit checkpoint. The app analyzes the clip and returns a structured JSON output indicating the presence or absence of events. The system is designed using a threaded architecture, with one thread responsible for capturing frames and another for running inference. These threads communicate via shared data structures, and thread-safe mechanisms like deque and threading.Event ensure consistent operation without requiring explicit locks. The entire workflow is orchestrated and visualized through a Streamlit-based dashboard that allows the user to select a video source, start and stop monitoring, view the live video feed, and check the detections in real time.

## Limitations

We run this on a Macbook Pro M1 and the MLX library. Even thouth the version of the model we use is optimized for Apple Silicon, the inference latency is still high, so the responsiveness is not that great. The model sometimes does not detect an activity even though it explains that it happened.

## Usage

```
streamlit run app.py
```

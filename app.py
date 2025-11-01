import streamlit as st
import cv2, os, tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --------------------------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------------------------
st.set_page_config(
    page_title="Weightlifting Biomechanics Lab",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to Page", [
    "Page 1 – Project Overview",
    "Page 2 – Background & Theory",
    "Page 3 – Experiment",
    "Page 4 – Sample Output & Explanation",
    "Page 5 – Results Dashboard"
])

# --------------------------------------------------------------------
# PAGE 1 – FRONT PAGE / INFO
# --------------------------------------------------------------------
if page.startswith("Page 1"):
    st.title("Lower-Body Athletic Gait & Movement Analysis")

    st.markdown("""
    ### SRM Institute of Science and Technology
    **College of Engineering and Technology**  
    **School of Bioengineering**  
    **Department of Biomedical Engineering**  
    SRM Nagar, Kattankulathur – 603203, Chengalpattu District, Tamil Nadu  

    **Academic Year:** 2025–26 (ODD: FT)  
    **Course Code & Title:** 21BMC401J – *Biomechanics*  
    **Year & Semester:** IV Year, VII Semester  

    ---
    **Submitted By:**  
    **Gayathri S. H**  
    B.Tech Biomedical Engineering  
    Email: *pcmjs.gayathri@gmail.com*  

    ---
    ### Project Abstract
    This project analyzes human motion using computer-vision techniques.  
    A YOLO-based pose-estimation model extracts biomechanical parameters such as
    stride length, cadence, gait speed, joint angles, and barbell trajectory
    from athlete videos.  
    The aim is to provide objective, data-driven insights for performance enhancement and technique evaluation.
    """)

    st.markdown("Use the sidebar to navigate to Background, Experiment, Sample Output, and Results pages.")

# --------------------------------------------------------------------
# PAGE 2 – BACKGROUND / THEORY
# --------------------------------------------------------------------
elif page.startswith("Page 2"):
    st.title("Background & Theory")

    st.header("Gait and Biomechanics Overview")
    st.write("""
    Gait refers to the pattern of limb movements during locomotion.
    A complete gait cycle starts at heel-strike and ends with the next heel-strike of the same foot.

    **Phases of Gait:**
    * **Stance Phase (~60%)** – Foot is in contact with the ground.
    * **Swing Phase (~40%)** – Foot moves forward to initiate the next step.
    """)

    st.header("Relevance of Gait and Motion Analysis")
    st.write("""
    * **Clinical Rehabilitation** – Evaluates postural control and mobility recovery.  
    * **Sports Biomechanics** – Optimizes athlete performance and prevents injuries.  
    * **Assistive Device Design** – Guides development of prosthetics and orthotics.
    """)

    st.header("Weightlifting Biomechanics")
    st.write("""
    In Olympic lifts such as the Snatch and Clean & Jerk, synchronized hip, knee,
    and shoulder motion enables efficient barbell movement.  
    Tracking these joint angles helps identify power generation and asymmetry patterns.
    """)

    st.header("Parameters Extracted")
    st.table(pd.DataFrame({
        "Parameter": [
            "Stride Length", "Cadence", "Gait Speed",
            "Joint Angles", "Barbell Trajectory", "Lifting Velocity"
        ],
        "Description": [
            "Distance between consecutive heel strikes",
            "Number of steps per minute",
            "Stride Length × Cadence / 120",
            "Angle between limb segments",
            "Path of the barbell centroid over time",
            "Rate of vertical barbell displacement (Δy/Δt)"
        ]
    }))

    st.header("Software Tools and Libraries")
    st.markdown("""
    * OpenCV – Video frame processing  
    * Ultralytics YOLOv8-Pose – Pose detection and keypoint extraction  
    * NumPy & Pandas – Numerical and tabular computations  
    * Matplotlib – Data visualization  
    * Streamlit – Interactive interface
    """)

# --------------------------------------------------------------------
# PAGE 3 – EXPERIMENT / INTERACTIVE LAB
# --------------------------------------------------------------------
elif page.startswith("Page 3"):
    st.title("Experiment – Weightlifting and Gait Analysis")

    st.markdown("Upload a video to analyze joint angles, barbell trajectory, and gait metrics.")

    def calculate_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    video_file = st.file_uploader("Upload video file", type=["mp4", "avi", "mov"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name
        st.video(video_file)

        if st.button("Run Analysis"):
            st.info("Running YOLOv8 Pose Estimation...")
            progress = st.progress(0)
            model = YOLO("yolov8n-pose.pt")

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx, angle_data, left_ankles, right_ankles = 0, [], [], []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame, verbose=False)
                if results[0].keypoints is not None:
                    kps = results[0].keypoints.xy[0].cpu().numpy()
                    L_HP,L_KN,L_AN,R_HP,R_KN,R_AN,L_WR,R_WR = 11,13,15,12,14,16,9,10
                    left_knee = calculate_angle(kps[L_HP], kps[L_KN], kps[L_AN])
                    right_knee = calculate_angle(kps[R_HP], kps[R_KN], kps[R_AN])
                    bar_x = (kps[L_WR][0]+kps[R_WR][0])/2
                    bar_y = (kps[L_WR][1]+kps[R_WR][1])/2
                    left_ankles.append(kps[L_AN]); right_ankles.append(kps[R_AN])
                    angle_data.append({"frame": frame_idx, "left_knee": left_knee,
                                       "right_knee": right_knee, "bar_y": bar_y})
                frame_idx += 1
                if frame_idx % 10 == 0:
                    progress.progress(min(frame_idx/total_frames,1.0))
            cap.release()
            progress.progress(1.0)

            if not angle_data:
                st.warning("No pose data detected. Try a clearer video.")
            else:
                df = pd.DataFrame(angle_data)
                df["time_sec"] = df["frame"]/fps
                df["bar_velocity"] = df["bar_y"].diff().fillna(0)*-1

                left_y = [a[1] for a in left_ankles if len(a)==2]
                stride_pix = np.max(left_y)-np.min(left_y) if len(left_y)>5 else 0
                stride_length = stride_pix/100
                peaks = np.sum(np.diff(np.sign(np.diff(left_y)))<0)
                cadence = (peaks/(len(df)/fps))*60 if len(df)>5 else 0
                gait_speed = stride_length*(cadence/120)

                summary = {
                    "Avg Left Knee Angle (°)": round(np.mean(df["left_knee"]),2),
                    "Avg Right Knee Angle (°)": round(np.mean(df["right_knee"]),2),
                    "Avg Bar Velocity": round(np.mean(df["bar_velocity"]),2),
                    "Stride Length (m)": round(stride_length,3),
                    "Cadence (steps/min)": round(cadence,2),
                    "Gait Speed (m/s)": round(gait_speed,3)
                }

                st.session_state["results_df"] = df
                st.session_state["summary"] = summary
                st.success("Analysis complete. Proceed to 'Sample Output & Explanation'.")

# --------------------------------------------------------------------
# PAGE 4 – SAMPLE OUTPUT & EXPLANATION
# --------------------------------------------------------------------
elif page.startswith("Page 4"):
    st.title("Sample Output and Explanation")

    st.write("""
    This section presents a sample of the expected analysis output.  
    When a video is processed, the app detects body keypoints and computes:
    * **Knee Angles** – Indicates flexion/extension coordination.
    * **Barbell Trajectory** – Helps evaluate lifting path efficiency.
    * **Stride Length & Cadence** – Estimate gait performance and rhythm.
    """)

    st.image("https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/pose-results.jpg",
             caption="Example of YOLOv8 Pose Detection Output")

    st.markdown("""
    **Interpretation Example:**
    * A **steady bar path** suggests efficient movement.
    * Large differences between **left and right knee angles** may indicate asymmetry.
    * The **velocity curve** reveals phases of acceleration and control during lifting.
    """)

    st.info("After understanding this example, proceed to Page 5 – Results Dashboard to view your own data.")

# --------------------------------------------------------------------
# PAGE 5 – RESULTS DASHBOARD
# --------------------------------------------------------------------
elif page.startswith("Page 5"):
    st.title("Results Dashboard")

    if "summary" not in st.session_state:
        st.warning("Please complete the experiment on Page 3 first.")
    else:
        summary = st.session_state["summary"]
        df = st.session_state["results_df"]

        st.subheader("Summary Metrics")
        st.dataframe(pd.DataFrame([summary]))

        st.subheader("Joint Angles and Bar Trajectory")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df["time_sec"], df["left_knee"], label="Left Knee Angle (°)")
        ax.plot(df["time_sec"], df["right_knee"], label="Right Knee Angle (°)")
        ax.plot(df["time_sec"], df["bar_y"]/5, label="Bar Height (scaled)")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle / Height")
        ax.grid(); ax.legend()
        st.pyplot(fig)

        st.subheader("Export Data")
        csv_full = df.to_csv(index=False).encode("utf-8")
        csv_summary = pd.DataFrame([summary]).to_csv(index=False).encode("utf-8")
        st.download_button("Download Full Metrics (CSV)", data=csv_full, file_name="full_metrics.csv")
        st.download_button("Download Summary Metrics (CSV)", data=csv_summary, file_name="summary_metrics.csv")

        st.markdown("""
        ---
        ### Performance Insights
        * High cadence with short stride → Faster but less stable gait  
        * Asymmetric knee angles → Possible imbalance  
        * Sudden velocity spikes → Indicate lift power phase
        ---
        """)

# --------------------------------------------------------------------
# END OF APP
# --------------------------------------------------------------------

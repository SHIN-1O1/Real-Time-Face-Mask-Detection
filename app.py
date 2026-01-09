"""
Face Mask Detection - Streamlit Web Application
Real-time detection using YOLOv8 with webcam and file upload support.
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import time

# Page configuration
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .stat-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .mask-green { color: #28a745; }
    .mask-red { color: #dc3545; }
    .mask-yellow { color: #ffc107; }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# Color scheme for detections
COLORS = {
    "with_mask": (0, 255, 0),       # Green
    "without_mask": (0, 0, 255),     # Red (BGR)
    "mask_weared_incorrect": (0, 255, 255)  # Yellow
}

CLASS_NAMES = ["with_mask", "without_mask", "mask_weared_incorrect"]
CLASS_LABELS = {
    "with_mask": "😷 Mask",
    "without_mask": "❌ No Mask",
    "mask_weared_incorrect": "⚠️ Incorrect"
}


@st.cache_resource
def load_model():
    """Load YOLOv8 model (cached for performance)."""
    from ultralytics import YOLO
    
    # Try multiple possible model paths
    model_paths = [
        Path(__file__).parent / "models" / "best.pt",
        Path(__file__).parent / "runs" / "detect" / "runs" / "detect" / "best_70ep_640px" / "weights" / "best.pt",
        Path(__file__).parent / "runs" / "detect" / "face_mask" / "weights" / "best.pt",
    ]
    
    for model_path in model_paths:
        if model_path.exists():
            print(f"Loading model from: {model_path}")
            return YOLO(str(model_path))
    
    # Fall back to pretrained YOLOv8 if custom model not found
    st.warning("⚠️ Custom model not found. Please train the model first using `python train.py`")
    st.info("Using pretrained YOLOv8n for demo (won't detect masks properly)")
    return YOLO("yolov8n.pt")


def draw_detections(frame, results, stats):
    """Draw bounding boxes and labels on frame."""
    annotated = frame.copy()
    
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class and confidence
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls_id >= len(CLASS_NAMES):
                continue
                
            class_name = CLASS_NAMES[cls_id]
            color = COLORS[class_name]
            label = f"{CLASS_LABELS[class_name]} {conf:.2f}"
            
            # Update stats
            stats[class_name] += 1
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - 30), (x1 + w + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1 + 5, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return annotated, stats


def process_image(image, model, conf_threshold):
    """Process a single image and return annotated result."""
    stats = {"with_mask": 0, "without_mask": 0, "mask_weared_incorrect": 0}
    
    # Run inference
    results = model.predict(image, conf=conf_threshold, verbose=False)
    
    # Draw detections
    annotated, stats = draw_detections(image, results, stats)
    
    return annotated, stats


def main():
    # Header
    st.markdown('<h1 class="main-header">😷 Face Mask Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Real-time detection using YOLOv8 Deep Learning</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    # Sidebar settings
    st.sidebar.markdown("## ⚙️ Settings")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🎨 Color Legend")
    st.sidebar.markdown("🟢 **Green** - Wearing Mask")
    st.sidebar.markdown("🔴 **Red** - No Mask")
    st.sidebar.markdown("🟡 **Yellow** - Incorrect Mask")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📷 Webcam", "🖼️ Image Upload", "🎬 Video Upload"])
    
    # Tab 1: Webcam
    with tab1:
        st.markdown("### Live Webcam Detection")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            run_webcam = st.checkbox("🎥 Start Webcam", key="webcam")
            frame_placeholder = st.empty()
        
        with col2:
            st.markdown("#### 📊 Live Stats")
            stats_placeholder = st.empty()
        
        if run_webcam:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Could not open webcam. Please check your camera connection.")
            else:
                fps_time = time.time()
                frame_count = 0
                
                while run_webcam:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠️ Failed to read frame from webcam")
                        break
                    
                    # Process frame
                    annotated, stats = process_image(frame, model, conf_threshold)
                    
                    # Calculate FPS
                    frame_count += 1
                    if time.time() - fps_time >= 1:
                        fps = frame_count
                        frame_count = 0
                        fps_time = time.time()
                    
                    # Convert BGR to RGB for display
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    
                    # Display frame
                    frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
                    
                    # Update stats
                    with stats_placeholder.container():
                        st.metric("😷 Mask", stats["with_mask"])
                        st.metric("❌ No Mask", stats["without_mask"])
                        st.metric("⚠️ Incorrect", stats["mask_weared_incorrect"])
                        st.metric("🎯 FPS", f"{fps if 'fps' in dir() else 0}")
                    
                    # Check if checkbox is still checked
                    if not st.session_state.get("webcam", False):
                        break
                
                cap.release()
    
    # Tab 2: Image Upload
    with tab2:
        st.markdown("### Upload an Image")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_upload")
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_np
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Original Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.markdown("#### Detection Result")
                
                # Process image
                annotated, stats = process_image(image_bgr, model, conf_threshold)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                
                st.image(annotated_rgb, use_container_width=True)
            
            # Stats display
            st.markdown("---")
            st.markdown("### 📊 Detection Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number mask-green">{stats['with_mask']}</div>
                    <div>😷 Wearing Mask</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number mask-red">{stats['without_mask']}</div>
                    <div>❌ No Mask</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number mask-yellow">{stats['mask_weared_incorrect']}</div>
                    <div>⚠️ Incorrect Mask</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                total = sum(stats.values())
                compliance = (stats['with_mask'] / total * 100) if total > 0 else 0
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number" style="color: {'#28a745' if compliance >= 80 else '#dc3545'}">{compliance:.1f}%</div>
                    <div>✅ Compliance Rate</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Tab 3: Video Upload
    with tab3:
        st.markdown("### Upload a Video")
        
        uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"], key="video_upload")
        
        if uploaded_video is not None:
            # Save uploaded video to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_video.read())
                video_path = tmp.name
            
            # Process video
            if st.button("🚀 Process Video"):
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                progress_bar = st.progress(0)
                frame_placeholder = st.empty()
                
                cumulative_stats = {"with_mask": 0, "without_mask": 0, "mask_weared_incorrect": 0}
                
                frame_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every 3rd frame for speed
                    if frame_idx % 3 == 0:
                        annotated, stats = process_image(frame, model, conf_threshold)
                        
                        # Update cumulative stats
                        for key in cumulative_stats:
                            cumulative_stats[key] += stats[key]
                        
                        # Display
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(annotated_rgb, use_container_width=True)
                    
                    # Update progress
                    frame_idx += 1
                    progress_bar.progress(frame_idx / total_frames)
                
                cap.release()
                
                st.success("✅ Video processing complete!")
                
                # Final stats
                st.markdown("### 📊 Video Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("😷 Total Mask Detections", cumulative_stats["with_mask"])
                with col2:
                    st.metric("❌ Total No Mask", cumulative_stats["without_mask"])
                with col3:
                    st.metric("⚠️ Total Incorrect", cumulative_stats["mask_weared_incorrect"])
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem;">
        <p>Built with ❤️ using YOLOv8 and Streamlit</p>
        <p>Face Mask Detection System - Real-time Safety Compliance</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

import streamlit as st
import tempfile
import os
import cv2
import pandas as pd
from utils.ui_helpers import apply_custom_css, render_header, render_metrics, aggregate_segments, format_time, create_chart_data
from core.identifier import FaceIdentifier
from core.processor import VideoProcessor

# Page Config
st.set_page_config(
    page_title="AI Video Face Search",
    page_icon="🔍",
    layout="wide"
)

# Apply Styling
apply_custom_css()

# Initialize Session State
if 'identifier' not in st.session_state:
    st.session_state.identifier = FaceIdentifier()
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'results' not in st.session_state:
    st.session_state.results = []
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = None

def main():
    render_header()

    # --- Step 1: Inputs ---
    st.markdown('<div class="step-header"><h3>Step 1: Data Selection</h3></div>', unsafe_allow_html=True)
    
    # Data Source Selection
    data_source = st.radio("Choose Data Source:", ["Upload Custom Data", "Use Demo Data (Obama)", "Use Demo Data (Merkel)"], horizontal=True)
    
    video_path = None
    image_inputs = []
    
    col_vid, col_img = st.columns([1, 1])
    
    # Logic for Inputs
    if "Demo" in data_source:
        demo_name = "obama" if "Obama" in data_source else "merkel"
        st.session_state.demo_mode = demo_name
        
        # Paths
        video_path = os.path.join(os.getcwd(), 'assets', 'videos', f'{demo_name}.mp4')
        base_img_path = os.path.join(os.getcwd(), 'assets', 'images', f'{demo_name}')
        image_inputs = [os.path.join(base_img_path, f) for f in os.listdir(base_img_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        with col_vid:
            st.subheader("Target Video")
            st.video(video_path)
            
        with col_img:
            st.subheader("Reference Images")
            # Show all images in a grid
            cols = st.columns(4)
            for i, img_path in enumerate(image_inputs):
                cols[i % 4].image(img_path, width='stretch')
            st.caption(f"Total {len(image_inputs)} reference images available.")
            
    else:
        st.session_state.demo_mode = None
        
        with col_vid:
            st.subheader("Target Video")
            uploaded_video = st.file_uploader("Upload Video (MP4/MOV)", type=['mp4', 'mov', 'avi'])
            if uploaded_video:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
                tfile.write(uploaded_video.read())
                tfile.close()
                video_path = tfile.name
                st.video(video_path)

        with col_img:
            st.subheader("Reference Images")
            uploaded_images = st.file_uploader("Upload Photos (JPG/PNG)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
            if uploaded_images:
                image_inputs = uploaded_images
                # Preview
                cols = st.columns(4)
                for i, img in enumerate(uploaded_images):
                    cols[i % 4].image(img, width='stretch')

    # Enrollment Action
    st.markdown("<br>", unsafe_allow_html=True)
    col_enrol_btn, _ = st.columns([1, 3])
    with col_enrol_btn:
        if st.button("Enroll on Images", type="primary", disabled=not image_inputs):
            with st.spinner("Learning face features..."):
                # Reset
                st.session_state.identifier = FaceIdentifier()
                result = st.session_state.identifier.enroll_person(image_inputs)
                
                if result['success']:
                    st.success(f"Enrolled successfully!")
                else:
                    st.error(result['message'])

    # --- Step 2: Configuration ---
    if st.session_state.identifier.is_enrolled and video_path:
        st.markdown('<div class="step-header"><h3>Step 2: Configuration & Analysis</h3></div>', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([1, 1, 1])
        
        with c1:
            process_every_n = st.slider(
                "Sampling Rate (Frames)", 
                min_value=1, 
                max_value=60, 
                value=10, 
                help="Controls how often the AI checks for the face. \n\n- **1**: Checks every single frame (Slowest, most accurate). \n- **30**: Checks once per second (Fastest). \n- **Default (10)**: Good balance."
            )
            
        with c2:
            resize_factor = st.slider(
                "Processing Resolution", 
                min_value=0.1, 
                max_value=1.0, 
                value=0.5, 
                help="Reduces the video size internally before AI processing to speed things up. \n\n- **1.0**: Full resolution (Slow). \n- **0.5**: Half resolution (4x faster). \n- **0.25**: Quarter resolution (Very fast, might miss small faces)."
            )
            
        with c3:
            st.markdown("<br>", unsafe_allow_html=True) # Spacer
            start_btn = st.button("Start Video Analysis", type="primary", width='stretch')

        if start_btn:
            processor = VideoProcessor(
                video_path, 
                resize_factor=resize_factor, 
                process_every_n_frames=process_every_n
            )
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(p):
                progress_bar.progress(p)
                status_text.text(f"Scanning video... {int(p*100)}%")

            with st.spinner("Analyzing video footage..."):
                results = processor.process_video(
                    st.session_state.identifier.identify,
                    progress_callback=update_progress
                )
            
            st.session_state.results = results
            st.session_state.processing_complete = True
            st.session_state.last_video_path = video_path # Store for report
            st.session_state.last_step = process_every_n # Store for aggregation
            st.success(f"Analysis Complete! Found {len(results)} matches.")

    # --- Step 3: Report ---
    if st.session_state.processing_complete and st.session_state.results:
        st.markdown('<div class="step-header"><h3>Step 3: Final Report</h3></div>', unsafe_allow_html=True)
        
        # Re-display video for context
        if st.session_state.get('last_video_path'):
             st.video(st.session_state.last_video_path)

        results = st.session_state.results
        step = st.session_state.get('last_step', 10)
        
        from utils.ui_helpers import aggregate_segments, format_time, create_chart_data
        
        segments = aggregate_segments(results, step)
        
        # Build Table
        table_data = []
        for i, seg in enumerate(segments):
            duration = seg['end_time'] - seg['start_time']
            table_data.append({
                'ID': i + 1,
                'Time Range': f"{format_time(seg['start_time'])} - {format_time(seg['end_time'])}",
                'Duration': f"{duration:.1f}s",
                'Frame Range': f"{seg['start_frame']} - {seg['end_frame']}"
            })
        
        df_table = pd.DataFrame(table_data)
        
        # Build Chart
        max_time = results[-1]['timestamp'] if results else 0
        chart_df = create_chart_data(results, max_time)

        col_table, col_chart = st.columns([1, 1])
        
        with col_table:
            st.subheader("Detections Log")
            st.dataframe(df_table, width='stretch', hide_index=True, height=400)
            
        with col_chart:
            st.subheader("Presence Timeline")
            st.bar_chart(chart_df, x='Time (s)', y='Detected', color='#00ff00', height=400)

if __name__ == "__main__":
    main()

import streamlit as st

def apply_custom_css():
    """
    Applies custom CSS to the Streamlit app for a premium look.
    """
    st.markdown("""
        <style>
        /* Main container styling */
        .stApp {
            background-color: #ffffff;
            color: #31333F;
        }
        
        /* Headings */
        h1, h2, h3 {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            color: #1a1a1a;
        }
        
        /* Custom buttons */
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        /* Step Headers */
        .step-header {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 2rem;
            margin-bottom: 1rem;
            border-left: 5px solid #ff4b4b;
        }
        </style>
    """, unsafe_allow_html=True)

def render_header():
    """
    Renders the main header and explanatory text.
    """
    st.title("AI Video Face Search")
    
    st.markdown("""
    ### About this Demo
    This application demonstrates the power of **Computer Vision** and **AI** in automating video analysis. 
    Originally designed to help media teams sift through large amounts of footage, this tool allows you to 
    rapidly find relevant clips of specific people for news production and storytelling.
    
    **How it works:**
    The system takes a **target video** and **reference images** of a person as input. It "enrolls" the person 
    by learning their unique facial features from the images. It then scans the video frame-by-frame to 
    create a targeted search index, identifying exactly when that person appears.
    
    ### The Technology
    Under the hood, this tool leverages **Facenet-PyTorch** (using Inception Resnet V1) for state-of-the-art 
    face recognition and **OpenCV** for high-performance video processing. It uses Deep Learning to generate 
    512-dimensional vector embeddings for faces, allowing for robust identification even across different 
    lighting conditions and angles. **Note:** The AI learns best when provided with reference images showing 
    the face from multiple perspectives and angles.

    ### About the Author
    This application was built by **Brian Jin**, an AI Applications Engineer.
    *   🌐 [Website](https://brianjin.eu/)
    *   💻 [GitHub](https://github.com/zer0dude)
    """)
    st.divider()

def render_metrics(total_frames, processed_frames, matches_found):
    """
    Renders real-time metrics during processing.
    """
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Frames", total_frames)
    with col2:
        st.metric("Processed", processed_frames)
    with col3:
        st.metric("Matches Found", matches_found)

def format_time(seconds):
    """Formats seconds into MM:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def aggregate_segments(results, process_every_n):
    """
    Aggregates individual frame matches into continuous time segments.
    A segment is broken if the gap between frames > process_every_n.
    """
    if not results:
        return []
    
    # Sort by frame index to be safe
    results = sorted(results, key=lambda x: x['frame_index'])
    
    segments = []
    current_start = results[0]
    last_frame = results[0]['frame_index']
    last_res = results[0]
    
    # We define continuity as: next_frame == last_frame + process_every_n
    # If we skipped a processing step (e.g. no match found), the chain breaks.
    step = process_every_n
    
    for res in results[1:]:
        frame = res['frame_index']
        
        if frame - last_frame <= step:
            # Continuous
            last_frame = frame
            last_res = res
        else:
            # Break in continuity -> Close segment
            segments.append({
                'start_frame': current_start['frame_index'],
                'end_frame': last_frame,
                'start_time': current_start['timestamp'],
                'end_time': last_res['timestamp']
            })
            current_start = res
            last_frame = frame
            last_res = res
            
    # Append final segment
    segments.append({
        'start_frame': current_start['frame_index'],
        'end_frame': last_frame,
        'start_time': current_start['timestamp'],
        'end_time': last_res['timestamp']
    })
    
    return segments

def create_chart_data(results, total_duration):
    """
    Creates a DataFrame for a timeline bar chart (1-second bins).
    """
    import pandas as pd
    
    # Identify all seconds where a match occurred
    match_seconds = set(int(r['timestamp']) for r in results)
    
    data = []
    # Iterate through every second of the video
    for sec in range(int(total_duration) + 1):
        data.append({
            'Time (s)': sec,
            'Detected': 1 if sec in match_seconds else 0
        })
        
    return pd.DataFrame(data)

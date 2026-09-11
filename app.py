import streamlit as st
import easyocr
from PIL import Image
import numpy as np
import io

# 1. Page Configuration
st.set_page_config(page_title="AI OCR Hub", layout="wide")
st.title("📄 AI Document Extraction Hub")
st.write("Upload an invoice or document to extract raw text using OCR instantly.")

# 2. Cache the OCR Model (Prevents reloading on every click)
@st.cache_resource
def load_ocr_model():
    # Downloads the English language parameters on first use
    return easyocr.Reader(['en'], gpu=False)

try:
    reader = load_ocr_model()
except Exception as e:
    st.error(f"Error loading AI Model: {e}. Please check your internet connection for the initial download.")

# 3. UI Layout (Two Columns: Left for Upload, Right for Results)
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Upload Section")
    uploaded_file = st.file_uploader("Choose an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Open and securely render the preview
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Document Preview", use_container_width=True)

with col2:
    st.subheader("⚡ AI Processing & Results")
    
    if uploaded_file is not None:
        if st.button("Run OCR Extraction", type="primary"):
            with st.spinner("Analyzing document structure and extracting text..."):
                try:
                    # FIX: Safely convert uploaded bytes into a standard RGB NumPy array
                    # This completely bypasses the EasyOCR "Invalid Input Type" error.
                    uploaded_file.seek(0) # Reset stream pointer
                    img_open = Image.open(uploaded_file).convert('RGB')
                    img_array = np.array(img_open)
                    
                    # Run the OCR Engine
                    result = reader.readtext(img_array, detail=0)
                    
                    # Join lines into readable paragraph format
                    full_text = "\n".join(result)
                    
                    if not full_text.strip():
                        full_text = "OCR finished, but no readable English text was found in the image."
                    
                    # Store text inside Streamlit session state so it stays visible across re-runs
                    st.session_state['extracted_text'] = full_text
                    st.success("Extraction Complete!")
                    
                except Exception as error:
                    st.error(f"An error occurred during extraction: {error}")
            
        # Display results if text has been extracted
        if 'extracted_text' in st.session_state:
            st.text_area("Extracted Raw Text", value=st.session_state['extracted_text'], height=250)
            
            # Allow the user to download the extracted text directly to their computer
            st.download_button(
                label="📥 Download Extracted Text (.txt)",
                data=st.session_state['extracted_text'],
                file_name=f"extracted_{uploaded_file.name.split('.')[0]}.txt",
                mime="text/plain"
            )
    else:
        st.info("Please upload an image document on the left panel to begin.")

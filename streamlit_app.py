import streamlit as st
from PIL import Image
from io import BytesIO
import json
import asyncio
from app import (
    gemini_response,
    gemini_image_response,
    extract_x,
    job_json_planner,
    job_unified
)

# Page config
st.set_page_config(page_title="Ad Generator", layout="wide")

# Initialize session state
if "job_json" not in st.session_state:
    st.session_state.job_json = None
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None
if "usage_stats" not in st.session_state:
    st.session_state.usage_stats = None

st.title("🎨 Ad Generator")

# Sidebar for inputs
with st.sidebar:
    st.header("Input Images")
    
    # Ad reference image (required)
    ad_reference = st.file_uploader(
        "Ad Reference Image *",
        type=["png", "jpg", "jpeg", "webp"],
        help="The reference ad image to use as a template"
    )
    
    # Product image (required)
    product_image = st.file_uploader(
        "Product Image *",
        type=["png", "jpg", "jpeg", "webp"],
        help="The product image to insert into the ad"
    )
    
    # Logo image (optional)
    logo_image = st.file_uploader(
        "Logo Image (Optional)",
        type=["png", "jpg", "jpeg", "webp"],
        help="Brand logo to replace in the ad"
    )
    
    st.header("Campaign Information")
    
    # Campaign info text area
    campaign_info = st.text_area(
        "Brand Info / Campaign Data / User Directions",
        height=200,
        help="Enter brand information, campaign details, and any specific directions"
    )

# Helper function to convert uploaded file to PIL Image
def upload_to_pil(uploaded_file):
    if uploaded_file is None:
        return None
    return Image.open(BytesIO(uploaded_file.read()))

# Helper function to run async functions in Streamlit
def run_async(coro):
    """Run async function - Streamlit handles this well with asyncio.run"""
    return asyncio.run(coro)

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.header("Step 1: Generate JSON Plan")
    
    if st.button("Generate JSON Plan", type="primary", use_container_width=True):
        if ad_reference is None or product_image is None:
            st.error("Please upload both Ad Reference Image and Product Image")
        elif not campaign_info.strip():
            st.error("Please enter campaign information")
        else:
            with st.spinner("Generating JSON plan..."):
                try:
                    # Convert uploads to PIL Images
                    ad_ref_pil = upload_to_pil(ad_reference)
                    product_pil = upload_to_pil(product_image)
                    logo_pil = upload_to_pil(logo_image) if logo_image else None
                    
                    # Prepare images list in order: [IMAGE REFERENCE], [BRAND LOGO], [BRAND PRODUCT]
                    images_list = [ad_ref_pil]
                    if logo_pil:
                        images_list.append(logo_pil)
                    images_list.append(product_pil)
                    
                    # Generate JSON
                    prompt = job_json_planner.format(campaign_info)
                    response_text, usage = run_async(
                        gemini_response(prompt, images_list)
                    )
                    
                    # Extract JSON
                    json_str = extract_x(response_text, "json")
                    job_json = json.loads(json_str)
                    
                    # Store in session state
                    st.session_state.job_json = job_json
                    st.session_state.usage_stats = usage
                    
                    st.success("JSON plan generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating JSON: {str(e)}")
                    st.exception(e)
    
    # Display and edit JSON fields
    if st.session_state.job_json is not None:
        st.header("Step 2: Edit JSON Plan")
        
        job = st.session_state.job_json
        
        # Editable text fields for each JSON key
        text_swap = st.text_area(
            "Text Swap Instructions",
            value=job.get("text_swap", ""),
            height=150,
            help="Instructions for text and logo replacement"
        )
        
        product_swap = st.text_area(
            "Product Swap Instructions",
            value=job.get("product_swap", ""),
            height=150,
            help="Instructions for product replacement"
        )
        
        edits = st.text_area(
            "Edit Instructions",
            value=job.get("edits", ""),
            height=150,
            help="Instructions for visual grading and styling"
        )
        
        # Update session state with edited values
        st.session_state.job_json = {
            "text_swap": text_swap,
            "product_swap": product_swap,
            "edits": edits
        }
        
        st.header("Step 3: Generate Final Image")
        
        if st.button("Generate Ad Image", type="primary", use_container_width=True):
            if ad_reference is None or product_image is None:
                st.error("Please upload both Ad Reference Image and Product Image")
            else:
                with st.spinner("Generating ad image... This may take a while..."):
                    try:
                        # Convert uploads to PIL Images
                        ad_ref_pil = upload_to_pil(ad_reference)
                        product_pil = upload_to_pil(product_image)
                        logo_pil = upload_to_pil(logo_image) if logo_image else None
                        
                        # Prepare images list in order: [IMAGE REFERENCE], [BRAND LOGO], [BRAND PRODUCT]
                        images_list = [ad_ref_pil]
                        if logo_pil:
                            images_list.append(logo_pil)
                        images_list.append(product_pil)
                        
                        # Get current job JSON
                        current_job = st.session_state.job_json
                        
                        # Generate image
                        prompt = job_unified.format(
                            campaign_info,  # CAMPAIGN DATA
                            current_job["product_swap"],  # GUIDELINES
                            current_job["edits"]  # EDIT INSTRUCTIONS
                        )
                        
                        image_bytes, usage = run_async(
                            gemini_image_response(prompt, images_list)
                        )
                        
                        # Store in session state
                        st.session_state.generated_image = image_bytes
                        st.session_state.usage_stats = usage
                        
                        st.success("Ad image generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error generating image: {str(e)}")
                        st.exception(e)
        
        # Show reference images below the edit form
        st.header("Reference Images")
        ref_col1, ref_col2, ref_col3 = st.columns(3)
        
        with ref_col1:
            if ad_reference:
                st.caption("Ad Reference")
                st.image(ad_reference, use_container_width=True)
        
        with ref_col2:
            if product_image:
                st.caption("Product Image")
                st.image(product_image, use_container_width=True)
        
        with ref_col3:
            if logo_image:
                st.caption("Logo")
                st.image(logo_image, use_container_width=True)

with col2:
    st.header("Generated Ad")
    
    # Show generated image
    if st.session_state.generated_image:
        st.image(st.session_state.generated_image, use_container_width=True)
        
        # Download button
        st.download_button(
            label="Download Generated Ad",
            data=st.session_state.generated_image,
            file_name="generated_ad.png",
            mime="image/png",
            use_container_width=True
        )
        
        # Show usage stats
        if st.session_state.usage_stats:
            with st.expander("Token Usage"):
                st.json(st.session_state.usage_stats)
    else:
        st.info("Generate an ad image to see it here")


import streamlit as st
import torch
from diffusers import FluxPipeline
import os
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="FLUX.1 Image Generator",
    page_icon="🎨",
    layout="wide"
)

# Set your Hugging Face token
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

def initialize_pipeline():
    """Initialize the FLUX.1 pipeline with proper settings."""
    try:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()  # Optimize memory usage
        return pipe
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

def generate_image(pipe, prompt, height, width, guidance_scale, num_steps, seed):
    """Generate an image using the FLUX.1 model."""
    try:
        generator = torch.Generator("cpu").manual_seed(seed)
        image = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            max_sequence_length=512,
            generator=generator
        ).images[0]
        return image
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

def main():
    st.title("🎨 FLUX.1 Image Generator")
    st.write("Generate images using the FLUX.1 model from Black Forest Labs")

    # Initialize the pipeline
    if 'pipeline' not in st.session_state:
        with st.spinner("Loading FLUX.1 model... This might take a few minutes."):
            st.session_state.pipeline = initialize_pipeline()

    # Create the sidebar for parameters
    st.sidebar.header("Generation Parameters")
    
    # Input parameters
    prompt = st.text_area("Enter your prompt:", 
                         height=100,
                         placeholder="Example: A cat holding a sign that says hello world")
    
    col1, col2 = st.columns(2)
    with col1:
        height = st.slider("Image Height", min_value=512, max_value=1024, value=1024, step=64)
        guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=3.5, step=0.5)
    
    with col2:
        width = st.slider("Image Width", min_value=512, max_value=1024, value=1024, step=64)
        num_steps = st.slider("Number of Steps", min_value=20, max_value=100, value=50, step=5)
    
    seed = st.number_input("Random Seed", min_value=0, max_value=999999, value=0)

    # Generate button
    if st.button("Generate Image"):
        if not prompt:
            st.warning("Please enter a prompt first.")
            return
        
        if st.session_state.pipeline is None:
            st.error("Model failed to initialize. Please refresh the page and try again.")
            return
        
        with st.spinner("Generating image... This might take a few minutes."):
            image = generate_image(
                st.session_state.pipeline,
                prompt,
                height,
                width,
                guidance_scale,
                num_steps,
                seed
            )
            
            if image:
                st.success("Image generated successfully!")
                st.image(image, caption=prompt, use_column_width=True)
                
                # Add download button
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                st.download_button(
                    label="Download Image",
                    data=buf.getvalue(),
                    file_name="generated_image.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()

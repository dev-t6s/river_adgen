from typing import List, Union
from PIL import Image
from io import BytesIO
import re
import json
from google import genai

import os 
import asyncio
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def extract_x(response: str, code_type: str) -> str:
    pattern = rf"```{code_type}\s*(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    return match.group(1).strip() if match else response

def load_safe_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as img:
            img.load()  # Force PIL to read the file data now
            # Create a copy to ensure it's in memory and detach from file handle
            return img.copy() 
    except Exception as e:
        print(f"❌ CRITICAL ERROR: The image at '{path}' is corrupt or broken.")
        print(f"Details: {e}")
        raise ValueError(f"Cannot proceed with broken image: {path}")


async def gemini_response(
    prompt: str, images: List[Union[bytes, Image.Image]] = [], model="gemini-3-pro-preview"
) -> tuple[str, dict]:
    """Returns (response_text, usage_dict) where usage_dict contains 'input_tokens' and 'output_tokens'"""
    pil_images = []
    if images:
        for image in images:
            if isinstance(image, bytes):
                pil_images.append(Image.open(BytesIO(image)))
            else:
                pil_images.append(image)
    
    def _call_gemini():
        response = gemini_client.models.generate_content(
            model=model,
            contents=[prompt] + pil_images)
        
        usage = {"input_tokens": 0, "output_tokens": 0}
        if hasattr(response, 'usage_metadata'):
            usage["input_tokens"] = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
            usage["output_tokens"] = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
        elif hasattr(response, 'usage'):
            usage["input_tokens"] = getattr(response.usage, 'prompt_token_count', 0) or 0
            usage["output_tokens"] = getattr(response.usage, 'candidates_token_count', 0) or 0
        
        return response.text, usage
    
    return await asyncio.to_thread(_call_gemini)

async def gemini_image_response(
    prompt: str, images: List[Union[bytes, Image.Image]] = [], model="gemini-3-pro-image-preview"
) -> tuple[bytes, dict]:
    """Returns (image_bytes, usage_dict) where usage_dict contains 'input_tokens' and 'output_tokens'"""
    pil_images = []
    if images:
        for image in images:
            if isinstance(image, bytes):
                pil_images.append(Image.open(BytesIO(image)))
            else:
                pil_images.append(image)
    
    def _call_gemini():
        response = gemini_client.models.generate_content(
            model=model,
            contents=[prompt] + pil_images,
            config={
                "response_modalities": ["IMAGE"],
                "image_config": {
                    "aspectRatio": "4:5",
                }
            })
        
        usage = {"input_tokens": 0, "output_tokens": 0}
        if hasattr(response, 'usage_metadata'):
            usage["input_tokens"] = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
            usage["output_tokens"] = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
        elif hasattr(response, 'usage'):
            usage["input_tokens"] = getattr(response.usage, 'prompt_token_count', 0) or 0
            usage["output_tokens"] = getattr(response.usage, 'candidates_token_count', 0) or 0
        
        # Get the image (assuming it's a PIL Image object)
        image_data = None
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_data = part.inline_data.data
                break
        
        return image_data, usage
    
    return await asyncio.to_thread(_call_gemini)

brand_info="""
7-10 (Seven-Ten) is a homegrown Indian sneaker brand founded by Shibani Bhagat that positions itself as a high-fashion yet affordable streetwear label for "oversteppers"—those who defy conventional boundaries.[1] Their "Karigar" campaign serves as a tribute to India's unseen builders and artisans ("Karigars"), aiming to shift the narrative from simply wearing a brand to carrying the legacy of the hands that create.[2] The collection features sneakers with culturally rooted colorways like Bombay Black, Kaali Peeli, and Agra White, symbolizing a movement that values dignity of labor and explicitly honors the workforce that keeps the country moving."""

job_json_planner = """
You are a Creative Director and Technical Ad Strategist.
I have provided an [IMAGE REFERENCE], [BRAND LOGO], [BRAND PRODUCT], and specific [CAMPAIGN DATA].

YOUR TASK:
Analyze the inputs and output a strategic JSON object that dictates exactly how to transform the reference image into a campaign ad.

OUTPUT FORMAT:
Return ONLY valid JSON. Do not include markdown formatting (like ```json). Use the exact keys below:
```json
{{
"text_swap": "<string>",
"product_swap": "<string>",
"edits": "<string>"
}}
```

GUIDELINES FOR EACH FIELD:

1. "text_swap": 
   - Analyze the text hierarchy in [IMAGE REFERENCE]. Map every existing text element (Headline, CTA, Subhead) to specific copy from [CAMPAIGN DATA].
   - LOGO: Give instructions to replace the existing logo with [BRAND LOGO], specifying position and contrast color (e.g., "Use White Logo version").
   - RULES: Only replace existing text areas. Do NOT add new text boxes. If [IMAGE REFERENCE] has no text, only instruct on Logo placement. Ignore text printed physically on the product.

2. "product_swap": 
   - detailed instructions to replace the subject in [IMAGE REFERENCE] with [BRAND PRODUCT].
   - ALIGNMENT: Explicitly state to match the exact Scale, Position, and 3D Rotation (Yaw/Pitch/Roll) of the original subject. ANY OTHER EXPLICIT INSTRUCTIONS THAT ARE NEEDED TO MAKE THE PRODUCT FIT THE REFERENCE IMAGE SHOULD BE INCLUDED.
   - MULTI-INSTANCE CHECK: If the reference shows multiple items (e.g., a pair of shoes, a row of products), instruct to replace EVERY instance with [BRAND PRODUCT].

3. "edits": 
   - Instructions for global visual grading to match the [CAMPAIGN DATA] vibe.
   - COLORS: Specify which brand colors to highlight or how to color-grade the background , BASICALLY EDITING THE OVERALL LOOK OF THE AD.
   - FONTS: Suggest font styles (e.g., "Change font to Bold Sans-Serif") that align with the campaign identity.

[CAMPAIGN DATA]:
{}
"""


job_json_planner_prop_swap = """
You are a Creative Director and Technical Ad Strategist.
I have provided an [IMAGE REFERENCE], [BRAND LOGO], [BRAND PRODUCT], and specific [CAMPAIGN DATA].

YOUR TASK:
Analyze the inputs and output a strategic JSON object that dictates exactly how to transform the reference image into a campaign ad.

OUTPUT FORMAT:
Return ONLY valid JSON. Do not include markdown formatting (like ```json). Use the exact keys below:
```json
{{
"text_swap": "<string>",
"product_swap": "<string>",
"edits": "<string>"
}}
```

GUIDELINES FOR EACH FIELD:

1. "text_swap": 
   - Analyze the text hierarchy in [IMAGE REFERENCE]. Map every existing text element (Headline, CTA, Subhead) to specific copy from [CAMPAIGN DATA].
   - LOGO: Give instructions to replace the existing logo with [BRAND LOGO], specifying position and contrast color (e.g., "Use White Logo version").
   - RULES: Do NOT add any new text boxes or text elements anywhere else in the image. Only work with existing text areas. If [IMAGE REFERENCE] has no text, only instruct on Logo placement. Ignore text printed physically on the product.

2. "product_swap": 
   - Use the [IMAGE REFERENCE] as the base skeleton - maintain the layout and composition structure, but change the overall visual sense and direction of the image. DO NOT CHANGE POSITION OF HUMAN (IF PRESENT)
   - Transform the visual direction by replacing props, surfaces, backgrounds, and visual elements with new ones that align with [CAMPAIGN DATA] and brand identity.
   - Provide instructions for positioning the [BRAND PRODUCT]: placement, scale, orientation/angle (use textual descriptions like "front-facing", "slightly tilted", "side view", "angled toward camera"), and depth relationship.
   - Suggest lighting style, surface materials, textures, and color palette that create a new visual direction while keeping the compositional skeleton.
   - Explicitly state how to match/change the scale, position, and orientation of the original subject to fit the new visual context using descriptive text (not numeric values).
   - If the reference shows multiple items, instruct to replace EVERY instance with [BRAND PRODUCT] while maintaining compositional rhythm.

3. "edits": 
   - Instructions for global visual grading to match the [CAMPAIGN DATA] vibe.
   - COLORS: Specify which brand colors to highlight or how to color-grade the background , BASICALLY EDITING THE OVERALL LOOK OF THE AD.
   - FONTS: Suggest font styles (e.g., "Change font to Bold Sans-Serif") that align with the campaign identity.

[CAMPAIGN DATA]:
{}
"""

job_unified = """
You are an Expert Professional Ad Photographer, Graphic Designer, Typographer, Product Retoucher, Layout Specialist, and AI Compositor.

I have provided:
- [AD IMAGE REFERENCE]: The reference ad image that serves as the strict layout and composition guide
- [BRAND LOGO]: The brand logo to replace the existing logo
- [BRAND PRODUCT]: The product image which must be the hero subject
- [CAMPAIGN DATA]: Text and messaging for the campaign
- [GUIDELINES]: Specific instructions for product swap (Scale, Position, 3D Rotation, multi-instance requirements)
- [EDIT INSTRUCTIONS]: Specific instructions for final visual grading, colors, fonts, and overall aesthetic

YOUR TASK:
Transform the [AD IMAGE REFERENCE] into a flawless, magazine-quality professional advertisement that looks like it was shot by a top-tier commercial photographer, performing all transformations in a single pass while following all provided instructions precisely.

STEPS:

PHASE 1: TEXT & LOGO REPLACEMENT
1. TEXT REPLACEMENT:
   - Identify the headline and body copy positions in the [AD IMAGE REFERENCE]
   - Remove the original text and replace it with the messaging from [CAMPAIGN DATA] (Headlines/Slogans)
   - Match the font weight and style of the reference initially, but be ready to apply font styling from [EDIT INSTRUCTIONS] in Phase 3
   - Only replace existing text areas. Do NOT add new text boxes. If [AD IMAGE REFERENCE] has no text, skip this step

2. LOGO REPLACEMENT:
   - Locate the existing brand logo in the [AD IMAGE REFERENCE]
   - Remove it and place the attached [BRAND LOGO] in the same position and relative scale
   - Note: Logo color adjustments will be applied in Phase 3 based on [EDIT INSTRUCTIONS]

PHASE 2: PRODUCT SWAP
3. READ & UNDERSTAND GUIDELINES:
   - Carefully review the [GUIDELINES] provided. These contain specific instructions about:
     * Scale, Position, and 3D Rotation (Yaw/Pitch/Roll) requirements
     * Multi-instance replacement instructions (if multiple products need to be replaced)
     * Any special alignment or transformation requirements
   - Follow these guidelines EXACTLY as specified

4. LAYOUT & COUNT ANALYSIS:
   - Scan the [AD IMAGE REFERENCE] to identify the number of product instances present
   - **CRITICAL:** If the [GUIDELINES] specify multiple instances or the reference shows multiple products (e.g., a pair of shoes, a grid, or a repeating pattern), you must replace ALL instances as instructed
   - Analyze the exact screen space (bounding box), rotation, and scale of the original products

5. GEOMETRIC ALIGNMENT & INSERTION (Following Guidelines):
   - Remove the original subjects from the [AD IMAGE REFERENCE]
   - Insert the [BRAND PRODUCT] following the [GUIDELINES] specifications:
     * Apply the exact Scale, Position, and 3D Rotation (Yaw/Pitch/Roll) as specified in [GUIDELINES]
     * If [GUIDELINES] contain additional transformation instructions, apply them precisely
   - The new product must occupy the exact same negative space and silhouette as the old one, or as specified in [GUIDELINES]

6. COMPOSITING & BLENDING:
   - Adjust the lighting on the [BRAND PRODUCT] to match the reference scene (e.g., if light comes from the left in the reference, light the new product from the left, if shadow comes from the right, cast shadow from the right)
   - Ensure seamless integration with the background and other elements
   - Make sure the product looks solid and realistic, not "pasted on"

PHASE 3: FINAL POLISH & PROFESSIONAL ENHANCEMENT
7. READ & UNDERSTAND EDIT INSTRUCTIONS:
   - Carefully review the [EDIT INSTRUCTIONS] provided. These contain specific guidance about:
     * Which brand colors to use for background, fonts, and logo
     * Font style recommendations
     * Color grading preferences
     * Overall visual aesthetic and mood
     * Any specific alignment or composition requirements
   - Follow these instructions EXACTLY as specified, while also applying professional standards

8. CHECK & REPLACE (CRITICAL):
   - Look closely at the image. Are there any "old" or "wrong" products that were missed?
   - If ANY original products remain, replace them immediately with the provided [BRAND PRODUCT]
   - Ensure EVERY single product instance in the scene matches the [BRAND PRODUCT]

9. ALIGNMENT & COMPOSITION FIXES:
   - Fix any misaligned text, logos, or graphic elements. Ensure perfect horizontal and vertical alignment
   - Check that all elements follow proper grid systems and visual hierarchy
   - Ensure the product is perfectly centered or positioned according to professional composition rules (rule of thirds, golden ratio, etc.)
   - Fix any crooked or tilted elements to create a polished, professional look
   - Apply any specific alignment requirements from [EDIT INSTRUCTIONS]

10. COLOR GRADING & ADJUSTMENTS (Following Edit Instructions):
    - BACKGROUND COLOR: 
      * Follow the [EDIT INSTRUCTIONS] for background color specifications
      * If [EDIT INSTRUCTIONS] specify brand colors, use those colors to create a cohesive, professional color palette
      * If not specified, adjust or replace the background color to create a sophisticated, on-brand palette that complements the product
    - FONT COLOR: 
      * Apply font color specifications from [EDIT INSTRUCTIONS]
      * Ensure all text has optimal contrast and readability
      * Adjust font colors to match the brand aesthetic and ensure they stand out clearly against backgrounds
    - LOGO COLOR: 
      * Follow [EDIT INSTRUCTIONS] for logo color specifications (e.g., "Use White Logo version", "Use Black Logo version")
      * Adjust logo colors to ensure brand consistency, proper contrast, and visibility
      * Make sure the logo looks crisp and professional
    - Overall color harmony: 
      * Apply color grading instructions from [EDIT INSTRUCTIONS]
      * Create a unified color scheme that looks like a professionally color-graded commercial photograph

11. TYPOGRAPHY & FONT STYLING (Following Edit Instructions):
    - Apply font style recommendations from [EDIT INSTRUCTIONS] (e.g., "Change font to Bold Sans-Serif", "Use modern minimalist typography")
    - Ensure all text is crisp, readable, and properly rendered
    - Fix any font rendering issues or blurry text
    - Make sure typography aligns with the campaign identity specified in [EDIT INSTRUCTIONS]

12. REPAIR GLITCHES & ARTIFACTS:
    - Fix any weird AI distortions, jagged edges, or bad blending seams
    - Remove any artifacts, noise, or compression issues
    - Smooth out any rough transitions or unnatural edges

13. PROFESSIONAL PHOTO QUALITY ENHANCEMENTS:
    - Sharpen the image (remove blur) while maintaining natural-looking detail
    - Enhance depth of field and focus to draw attention to the product
    - Adjust lighting to create a professional studio-quality look with proper highlights and shadows
    - Add subtle professional touches like soft vignetting, color grading, and contrast adjustments
    - Ensure the overall image has the polished, high-end aesthetic of luxury brand advertisements
    - Apply any specific visual mood or aesthetic from [EDIT INSTRUCTIONS]

14. GRAPHIC ELEMENTS:
    - Make sure graphic elements and logos are sharp and professional
    - Apply any graphic styling requirements from [EDIT INSTRUCTIONS]

OUTPUT:
A flawless, high-resolution, magazine-quality professional advertisement where:
- 100% of products are correct and match the [BRAND PRODUCT] and [BRAND LOGO]
- All text has been replaced with [CAMPAIGN DATA] messaging
- All elements are perfectly aligned and composed
- Colors follow the [EDIT INSTRUCTIONS] specifications and are professionally graded and harmonious
- Typography and fonts match the [EDIT INSTRUCTIONS] recommendations
- The image looks like it was shot by a top commercial photographer
- Every detail is crisp, polished, and ready for publication
- All [GUIDELINES] and [EDIT INSTRUCTIONS] have been precisely followed

[CAMPAIGN DATA]:
{}

[GUIDELINES]:
{}

[EDIT INSTRUCTIONS]:
{}
"""

async def flow(ad_reference_image, brand_info, brand_logo, product_image):
    # Track token usage across all calls
    total_input_tokens = 0
    total_output_tokens = 0
    
    output_0, usage_0 = await gemini_response(job_json_planner.format(brand_info), [ad_reference_image, brand_logo, product_image])
    total_input_tokens += usage_0.get("input_tokens", 0)
    total_output_tokens += usage_0.get("output_tokens", 0)
    
    job = json.loads(extract_x(output_0, "json"))
    
    output, usage_1 = await gemini_image_response(
        job_unified.format(job['text_swap'], job['product_swap'], job['edits']), 
        [ad_reference_image, brand_logo, product_image]
    )
    total_input_tokens += usage_1.get("input_tokens", 0)
    total_output_tokens += usage_1.get("output_tokens", 0)

    return output, {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens}


async def process_single_image(ref_image_path, brand_info, brand_logo, product_image, output_dir, semaphore):
    async with semaphore:
        print(f"Processing: {ref_image_path.name}")
        
        # Load reference image
        ad_reference_image = load_safe_image(ref_image_path)
        
        # Generate ad
        output_image_bytes, usage = await flow(ad_reference_image, brand_info, brand_logo, product_image)
        
        # Save output
        output_filename = ref_image_path.stem + "_new_1_gen.png"
        output_path = output_dir / output_filename
        
        with open(output_path, "wb") as f:
            f.write(output_image_bytes)
        
        print(f"Saved: {output_path}")
        return usage

async def main():
    # Load shared assets
    brand_logo = load_safe_image(Path("./data/logo.png"))
    product_image = load_safe_image(Path("./data/image.png"))
    
    # Get all reference images
    references_dir = Path("./references")
    output_dir = Path("./output_new_1")
    output_dir.mkdir(exist_ok=True)
    
    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    
    # Collect all reference image paths
    ref_image_paths = [
        ref_image_path 
        for ref_image_path in references_dir.iterdir()
        if ref_image_path.suffix.lower() in image_extensions
    ]
    
    # Create semaphore to limit concurrent processing to 3
    semaphore = asyncio.Semaphore(3)
    
    # Process all images concurrently (max 3 at a time)
    tasks = [
        process_single_image(ref_image_path, brand_info, brand_logo, product_image, output_dir, semaphore)
        for ref_image_path in ref_image_paths[3:]
    ]
    
    usage_results = await asyncio.gather(*tasks)
    
    # Calculate averages
    if usage_results:
        total_input = sum(u.get("input_tokens", 0) for u in usage_results)
        total_output = sum(u.get("output_tokens", 0) for u in usage_results)
        count = len(usage_results)
        
        avg_input = total_input / count if count > 0 else 0
        avg_output = total_output / count if count > 0 else 0
        
        print("\n" + "="*50)
        print("TOKEN USAGE STATISTICS")
        print("="*50)
        print(f"Total images processed: {count}")
        print(f"Average input tokens: {avg_input:.2f}")
        print(f"Average output tokens: {avg_output:.2f}")
        print(f"Total input tokens: {total_input}")
        print(f"Total output tokens: {total_output}")
        print("="*50)
    
    print("All ads generated successfully!")


if __name__ == "__main__":
    asyncio.run(main())
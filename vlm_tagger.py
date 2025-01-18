import os
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
import itertools
import sys
from tqdm import tqdm
import time
from datetime import datetime

# Define supported image formats
SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.gif')

# Define a function for loading the processor and model
def load_model_and_processor(model_name, resolution_scale, device):
    """
    Load the processor and model with specified configurations.
    
    Args:
        model_name (str): The model's name on HuggingFace.
        resolution_scale (int): Scaling factor for image resolution.
        device (str): Device to load the model onto ('cuda' or 'cpu').
    
    Returns:
        processor, model: Loaded processor and model instances.
    """
    # Load the processor
    processor = AutoProcessor.from_pretrained(
        model_name, 
        size={"longest_edge": resolution_scale * 384}
    )
    
    # Set attention implementation based on device
    attn_impl = "flash_attention_2" if device == "cuda" else "eager"
    
    # Load the model
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Efficient precision for model weights
        _attn_implementation=attn_impl
    ).to(device)
    
    return processor, model


def process_image_directory(directory):
    """
    Process images in the given directory, supporting multiple formats
    (PNG, JPG, TIF, GIF) and generating individual text descriptions.
    
    Args:
        directory (str): Path to the directory containing images
    """
    # Start timing
    start_time = time.time()
    
    # Initialize counters
    successful_processes = 0
    failed_processes = 0
    error_log = []
    format_counts = {fmt: 0 for fmt in SUPPORTED_FORMATS}

    # Validate directory
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)


    # Usage
    MODEL_NAME = "HuggingFaceTB/SmolVLM-Instruct"
    RESOLUTION_SCALE = 4  # default value
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    processor, model = load_model_and_processor(MODEL_NAME, RESOLUTION_SCALE, DEVICE)

    # Find all supported image files in the directory
    image_files = []
    for filename in os.listdir(directory):
        lower_filename = filename.lower()
        for fmt in SUPPORTED_FORMATS:
            if lower_filename.endswith(fmt):
                image_files.append(filename)
                format_counts[fmt] += 1
                break

    total_images = len(image_files)
    
    if total_images == 0:
        print(f"No supported images found in the directory.")
        print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return
    
    print(f"\nFound {total_images} images to process:")
    for fmt, count in format_counts.items():
        if count > 0:
            print(f"- {fmt}: {count} files")
    
    # Process each image with progress bar
    for image_filename in tqdm(image_files, desc="Processing images", unit="image"):
        # Full path to image
        image_path = os.path.join(directory, image_filename)
        
        # Create output filename with tagged_ prefix and .txt suffix
        output_filename = f"tagged_{os.path.splitext(image_filename)[0]}.txt"
        output_path = os.path.join(directory, output_filename)
        
        # Load and process image
        try:
            image = load_image(image_path)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this image."}
                    ]
                },
            ]

            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[image], return_tensors="pt")
            inputs = inputs.to(DEVICE)

            generated_ids = model.generate(**inputs, max_new_tokens=500)
            full_decoded_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            delimiter = "Assistant: "
            if delimiter in full_decoded_output:
                response_text = full_decoded_output.split(delimiter, 1)[1].strip()
            else:
                response_text = "Error: Could not extract response."

            # Save response to individual text file
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(response_text)
            
            successful_processes += 1
            
        except Exception as e:
            error_message = f"Error processing {image_filename}: {str(e)}"
            error_log.append(error_message)
            failed_processes += 1

    # Calculate timing and prepare summary
    end_time = time.time()
    total_time = end_time - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    # Print summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total images found: {total_images}")
    print("\nBreakdown by format:")
    for fmt, count in format_counts.items():
        if count > 0:
            print(f"- {fmt}: {count} files")
    
    print(f"\nSuccessfully processed: {successful_processes}")
    print(f"Failed to process: {failed_processes}")
    print(f"Total processing time: {minutes} minutes, {seconds} seconds")
    if successful_processes > 0:
        print(f"Average time per image: {total_time/successful_processes:.2f} seconds")
    
    if error_log:
        print("\nErrors encountered:")
        for error in error_log:
            print(f"- {error}")
    
    print("="*50)

def main():
    # Check if directory is provided as an argument
    if len(sys.argv) != 2:
        print("Usage: python vlm_tagger.py <directory_path>")
        sys.exit(1)
    directory = sys.argv[1]
    process_image_directory(directory)

if __name__ == "__main__":
    main()
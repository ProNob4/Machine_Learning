import os
from PIL import Image
import pyheif
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Ensure pyheif is installed (Run this if needed)
!pip install pillow pyheif --quiet

def convert_heic_to_jpg(input_folder, output_folder):
    if not os.path.exists(input_folder):
        print(f"❌ Error: Input folder '{input_folder}' does not exist.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"✅ Created output folder: {output_folder}")

    files_converted = 0

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".heic"):  # Process only HEIC files
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".jpg")

            try:
                print(f"🔄 Processing: {input_path}")

                # Read HEIC file
                heif_file = pyheif.read(input_path)
                image = Image.frombytes(
                    heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode, heif_file.stride
                )

                # Save as JPG
                image.save(output_path, "JPEG")
                print(f"✅ Converted: {input_path} -> {output_path}")
                files_converted += 1

            except Exception as e:
                print(f"❌ Error processing file {input_path}: {e}")

    if files_converted == 0:
        print("⚠️ No HEIC files found in the input folder.")
    else:
        print(f"🎉 Conversion complete! {files_converted} files converted.")

# Define input and output directories
input_dir = "/content/drive/MyDrive/Data/Good Leaf"
output_dir = "/content/drive/MyDrive/Data/NGF"

convert_heic_to_jpg(input_dir, output_dir)

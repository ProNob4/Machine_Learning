{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOQxp+y5XICTc+BtlflL32W",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ProNob4/Machine_Learning/blob/main/Imagesizeconvert.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "import cv2\n",
        "from google.colab import drive\n",
        "\n",
        "# Mount Google Drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "def compress_and_resize_images(input_folder, output_folder, target_size=(224, 224), jpeg_quality=75):\n",
        "    # Create output folder if it doesn't exist\n",
        "    os.makedirs(output_folder, exist_ok=True)\n",
        "\n",
        "    for filename in os.listdir(input_folder):\n",
        "        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):\n",
        "            img_path = os.path.join(input_folder, filename)\n",
        "            try:\n",
        "                image = cv2.imread(img_path)\n",
        "                if image is None:\n",
        "                    print(f\"Failed to load {filename}. Skipping...\")\n",
        "                    continue\n",
        "\n",
        "                # Resize to target size\n",
        "                resized_image = cv2.resize(image, target_size)\n",
        "\n",
        "                # Set output path\n",
        "                output_filename = os.path.splitext(filename)[0] + '.jpg'  # Save all as .jpg\n",
        "                output_path = os.path.join(output_folder, output_filename)\n",
        "\n",
        "                # Save as compressed JPEG\n",
        "                cv2.imwrite(output_path, resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])\n",
        "\n",
        "                print(f\"Saved compressed: {output_filename}\")\n",
        "            except Exception as e:\n",
        "                print(f\"Error processing {filename}: {e}\")\n",
        "\n",
        "# Set folders\n",
        "input_folder = '/content/drive/MyDrive/DBML_output/Augmented GT'\n",
        "output_folder = '/content/drive/MyDrive/DBML_output/Compressed GT'\n",
        "\n",
        "# Compress & Resize\n",
        "compress_and_resize_images(input_folder, output_folder, target_size=(224, 224), jpeg_quality=75)\n"
      ],
      "metadata": {
        "id": "sv8YoTJKBCZU"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}

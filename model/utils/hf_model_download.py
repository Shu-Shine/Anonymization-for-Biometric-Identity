
from transformers import AutoModelForImageClassification, AutoConfig, AutoFeatureExtractor,AutoImageProcessor

base_model_name= "facebook/dinov2-base" 
fine_tuned_model_name = "Hemg/Wound-Image-classification"


try:
    # Try loading the processor for the fine-tuned model first
    processor = AutoImageProcessor.from_pretrained(fine_tuned_model_name)
    print(f"Successfully loaded image processor directly from {fine_tuned_model_name}")
except ValueError as e:
    processor = AutoImageProcessor.from_pretrained(base_model_name)
    print(f"Successfully loaded image processor from {base_model_name}")

model = AutoModelForImageClassification.from_pretrained(fine_tuned_model_name)
print(f"Successfully loaded model weights from {fine_tuned_model_name}")

# todo: Set a absolute path for saving
SHARED_STORAGE_PATH = "path/to/save/model_and_processor"


processor.save_pretrained(SHARED_STORAGE_PATH)
model.save_pretrained(SHARED_STORAGE_PATH)
print(f"Model and processor saved.")
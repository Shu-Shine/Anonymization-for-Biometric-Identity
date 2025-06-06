
MODEL_DICT = {
    # ViT
    "vit-b16-224-in21k": "google/vit-base-patch16-224-in21k",
    "vit-b32-224-in21k": "google/vit-base-patch32-224-in21k",
    "vit-l32-224-in21k": "google/vit-large-patch32-224-in21k",
    "vit-l15-224-in21k": "google/vit-large-patch16-224-in21k",
    "vit-h14-224-in21k": "google/vit-huge-patch14-224-in21k",
    "vit-b16-224": "google/vit-base-patch16-224",
    "vit-l16-224": "google/vit-large-patch16-224",
    "vit-b16-384": "google/vit-base-patch16-384",
    "vit-b32-384": "google/vit-base-patch32-384",
    "vit-l16-384": "google/vit-large-patch16-384",
    "vit-l32-384": "google/vit-large-patch32-384",
    # DINO
    "vit-b16-224-dino": "facebook/dino-vitb16",
    "vit-b8-224-dino": "facebook/dino-vitb8",
    "vit-s16-224-dino": "facebook/dino-vits16",
    "vit-s8-224-dino": "facebook/dino-vits8",
    # BEiT
    "beit-b16-224-in21k": "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "beit-l16-224-in21k": "microsoft/beit-large-patch16-224-pt22k-ft22k",
    # DINO v2
    "vit-s16-224-dino-v2": "facebook/dinov2-small",  # ViT-Small, Patch 16
    "vit-b16-224-dino-v2": "facebook/dinov2-base",  # ViT-Base, Patch 16
    "vit-l16-224-dino-v2": "facebook/dinov2-large",  # ViT-Large, Patch 16
    "vit-h14-224-dino-v2": "facebook/dinov2-giant",  # ViT-Huge/Giant, Patch 14

    # With finetuned weights
    # dinov2 models with fine-tuned weights
    "dinov2-base-skindisease": "Jayanth2002/dinov2-base-finetuned-SkinDisease",
    # google/vit-base-patch16-224-in21k with fine-tuned weights, 10 classes
    "vit-b16-224-in21k-wound": "Hemg/Wound-Image-classification",

    # --- CNNs from Torchvision ---
    # tag torchvision models with 'tv-' prefix
    "tv-vgg16": "torchvision/vgg16",
    "tv-vgg16-bn": "torchvision/vgg16_bn",  # VGG16 with BatchNorm
    "tv-resnet18": "torchvision/resnet18",
    "tv-resnet50": "torchvision/resnet50",
    "tv-efficientnet-b0": "torchvision/efficientnet_b0",
    "tv-efficientnet-b3": "torchvision/efficientnet_b3",
    "tv-efficientnet-b4": "torchvision/efficientnet_b4",

    # --- CNNs from Hugging Face Hub (Transformers) ---
    "hf-efficientnet-b0": "google/efficientnet-b0",
    # https://huggingface.co/models?pipeline_tag=image-classification
    # TIMM models on HF
    "timm-efficientnet-b0": "timm/efficientnet_b0.ra_in1k"  # Note: '.' in timm name often becomes '_'
}
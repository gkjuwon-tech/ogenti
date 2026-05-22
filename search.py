from huggingface_hub import HfApi
for m in HfApi().list_models(search="Wan-AI/Wan", limit=30):
    if "Diffusers" in m.id:
        print(m.id)
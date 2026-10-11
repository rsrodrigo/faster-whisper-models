from faster_whisper import WhisperModel

for model_name in [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3"
]:
    print(f"\nBaixando {model_name}...\n")
    WhisperModel(model_name, device="cpu", compute_type="int8")
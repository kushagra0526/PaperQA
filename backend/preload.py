"""Pre-downloads model weights during build step so runtime starts in 0.01s."""
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

if __name__ == "__main__":
    model_name = "deepset/minilm-uncased-squad2"
    print(f"Pre-caching {model_name} on disk...")
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForQuestionAnswering.from_pretrained(model_name)
    print("Model pre-cached successfully.")

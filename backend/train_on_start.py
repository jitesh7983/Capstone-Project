# train_on_start.py
import os

# Only train if model doesn't exist
if not os.path.exists("model.pkl"):
    print("Training models...")
    import train_model
    print("Training complete.")
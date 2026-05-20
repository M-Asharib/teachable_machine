import pickle
import os

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

print("==========================================")
print("     TEACHABLE MACHINE PRO MODEL INSPECTOR")
print("==========================================\n")

if not os.path.exists(model_path):
    print(f"❌ Error: Model file not found at '{model_path}'. Please train a model first.")
else:
    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        print(f"✅ Successfully loaded model from: {model_path}\n")
        print("--- Structure Keys ---")
        print(list(model_data.keys()))
        
        print("\n--- Feature Extractor Backbone ---")
        print(model_data.get("backbone_name", "Not specified (Defaulting to MobileNetV3)"))
        
        print("\n--- Category Mapping (Index -> Label) ---")
        print(model_data.get("label_map"))
        
        print("\n--- Classifier Configuration ---")
        clf = model_data.get("classifier")
        print(clf)
        if clf:
            print(f"  - Classes: {clf.classes_}")
            print(f"  - Features expected: {clf.n_features_in_}")
            print(f"  - Regularization Penalty: {getattr(clf, 'penalty', 'l2')}")
            print(f"  - Regularization Strength C: {getattr(clf, 'C', 1.0)}")
            
    except Exception as e:
        print(f"❌ Failed to parse pickle file: {str(e)}")

print("\n==========================================")

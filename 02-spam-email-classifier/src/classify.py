import sys
import argparse
import joblib

def main():
    parser = argparse.ArgumentParser(description="Classify text as Spam or Ham.")
    parser.add_argument("--text", type=str, required=True, help="Email or message text to classify")
    args = parser.parse_args()

    model_path = "models/spam_model.pkl"
    vectorizer_path = "models/vectorizer.pkl"

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    except Exception as e:
        print("Error loading model files. Make sure you ran spam_classifier.ipynb first.")
        sys.exit(1)

    # Transform input text and predict
    vectorized_text = vectorizer.transform([args.text])
    prediction = model.predict(vectorized_text)[0]
    probabilities = model.predict_proba(vectorized_text)[0]

    label = "SPAM" if prediction == 1 else "HAM (Legitimate)"
    confidence = max(probabilities) * 100

    print(f"\nResult: {label}")
    print(f"Confidence: {confidence:.2f}%\n")

if __name__ == "__main__":
    main()
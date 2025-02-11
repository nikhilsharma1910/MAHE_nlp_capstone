import joblib
from utils import preprocess_text

loaded_model = joblib.load('model_files/flu_detection_log_reg_model.pkl')
loaded_vectorizer = joblib.load('model_files/flu_vectorizer.pkl')


# Interactive loop for user input
print("🤖 Welcome to the Flu Detection System!")
print("Type your symptoms or any text, and I'll predict if it's flu-related.")
print("Type 'exit' to end the program.\n")


if __name__ == "__main__":
    while True:
        # Get user input
        user_input = input("Enter your message: ")

        # Exit condition
        if user_input.lower() == 'exit':
            print("Thanks for using the Flu Detection System. Stay healthy! 🌟")
            break

        cleaned_text = preprocess_text(user_input)
        X_new = loaded_vectorizer.transform([cleaned_text]).toarray()
        prediction = loaded_model.predict(X_new)[0]
        label = "Flu Detected! Please consider seeing a doctor." if prediction == 1 else "✅ No Flu Detected."
        print(f"Prediction: {label}\n")

import pandas as pd
import random
from datetime import datetime, timedelta

# Extended keyword sets
flu_keywords = [
    "fever", "cough", "chills", "sore throat", "body ache",
    "fatigue", "headache", "nausea", "runny nose", "congestn",
    "tired af", "sickkk", "😷", "ugh feeling bad", "sneeezing",
    "not sure if it's flu", "just exhausted", "allergies maybe?", "overthinking", "meh"
]
non_flu_keywords = [
    "party", "workout", "vacation", "travel", "birthday",
    "study grind", "meeting", "brunch", "netflix binge", "💯",
    "coffee time", "grind mode", "just chilling", "🔥", "lol tired",
    "stressed but not sick", "mood swings", "sleep deprived", "is this flu?", "idk anymore"
]
noise_words = [
    "random", "text", "just", "maybe", "idk", "so",
    "bruh", "lol", "omg", "nothing", "seriously", "whatever"
]

# Generate ambiguous texts
def generate_text(label):
    keywords = flu_keywords if label == "flu" else non_flu_keywords
    text = random.choices(keywords, k=random.randint(2, 5)) + random.choices(noise_words, k=random.randint(2, 6))
    random.shuffle(text)
    return " ".join(text)


data = []
start_date = datetime(2023, 1, 1)
for _ in range(2000):
    label = random.choice(["flu", "non-flu"])
    date = start_date + timedelta(days=random.randint(0, 365))
    text = generate_text(label)
    data.append({"Date": date.strftime("%Y-%m-%d"), "Text": text, "Label": label})


df = pd.DataFrame(data)
df.to_csv("flu_social_media_dataset.csv", index=False)

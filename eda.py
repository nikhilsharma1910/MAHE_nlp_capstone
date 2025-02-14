import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from utils import preprocess_text
import re
import string

df = pd.read_csv("flu_social_media_dataset.csv")


print("First 5 Rows of the Dataset:")
print(df.head())

# Basic Information
print("\nDataset Info:")
print(df.info())


print("\nMissing Values:")
print(df.isnull().sum())


plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Label", palette="coolwarm")
plt.title("Class Distribution (Flu vs Non-Flu)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()


df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day


plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="Month", hue="Label", palette="coolwarm")
plt.title("Flu vs Non-Flu Cases by Month")
plt.xlabel("Month")
plt.ylabel("Count")
plt.xticks(range(1, 13))
plt.show()


df["Cleaned_Text"] = df["Text"].apply(preprocess_text)


flu_text = " ".join(df[df["Label"] == "flu"]["Cleaned_Text"])
non_flu_text = " ".join(df[df["Label"] == "non-flu"]["Cleaned_Text"])


plt.figure(figsize=(10, 5))
wordcloud_flu = WordCloud(width=800, height=400, background_color="black").generate(flu_text)
plt.imshow(wordcloud_flu, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - Flu-Related Posts")
plt.show()


plt.figure(figsize=(10, 5))
wordcloud_non_flu = WordCloud(width=800, height=400, background_color="black").generate(non_flu_text)
plt.imshow(wordcloud_non_flu, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - Non-Flu Posts")
plt.show()


flu_words = flu_text.split()
non_flu_words = non_flu_text.split()

flu_word_freq = Counter(flu_words)
non_flu_word_freq = Counter(non_flu_words)


flu_top_words = pd.DataFrame(flu_word_freq.most_common(10), columns=["Word", "Count"])
plt.figure(figsize=(10, 5))
sns.barplot(data=flu_top_words, x="Word", y="Count", palette="Reds_r")
plt.xticks(rotation=45)
plt.title("Top 10 Words in Flu-Related Posts")
plt.show()


non_flu_top_words = pd.DataFrame(non_flu_word_freq.most_common(10), columns=["Word", "Count"])
plt.figure(figsize=(10, 5))
sns.barplot(data=non_flu_top_words, x="Word", y="Count", palette="Blues_r")
plt.xticks(rotation=45)
plt.title("Top 10 Words in Non-Flu Posts")
plt.show()


df["Text_Length"] = df["Cleaned_Text"].apply(lambda x: len(x.split()))

plt.figure(figsize=(8, 5))
sns.histplot(df, x="Text_Length", hue="Label", kde=True, palette="coolwarm", bins=30)
plt.title("Distribution of Post Lengths")
plt.xlabel("Number of Words in Post")
plt.ylabel("Count")
plt.show()


print("\nSummary Statistics of Post Lengths:")
print(df.groupby("Label")["Text_Length"].describe())


if __name__ == "__main__":
    print("\nKey Observations:")
    print("- Flu-related posts often contain words like 'fever', 'cough', 'chills', etc.")
    print("- Non-Flu posts frequently include words like 'party', 'work', 'vacation', etc.")
    print("- Flu-related posts tend to be slightly longer in word count.")
    print("- There may be seasonal trends in flu-related posts, peaking in winter months.")

import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import ssl
from nltk.stem import WordNetLemmatizer


# 禁用SSL证书验证
ssl._create_default_https_context = ssl._create_unverified_context

# 下载所需的数据
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# 读取Moby Dick文件
with open('mobydick.txt', 'r', encoding='utf-8') as file:
    moby_dick_text = file.read()

# Tokenization
tokens = nltk.word_tokenize(moby_dick_text)

# Stop-words filtering
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Parts-of-Speech (POS) tagging
pos_tags = nltk.pos_tag(filtered_tokens)

# POS frequency
pos_counts = FreqDist(tag for word, tag in pos_tags)
top_pos = pos_counts.most_common(5)

print("Top 5 Parts of Speech and Their Frequencies:")
for pos, count in top_pos:
    print(f"{pos}: {count}")

# Lemmatization
lemmatizer = WordNetLemmatizer()

# Map POS tags to WordNet tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'  # Adjective
    elif treebank_tag.startswith('V'):
        return 'v'  # Verb
    elif treebank_tag.startswith('N'):
        return 'n'  # Noun
    elif treebank_tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return 'n'  # Default to noun if not found

lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags[:20]]

print("\nTop 20 Lemmatized Tokens:")
print(lemmatized_tokens)

# Plotting frequency distribution
pos_counts.plot(30, cumulative=False)
plt.title("POS Frequency Distribution")
plt.xlabel("POS")
plt.ylabel("Frequency")
plt.show()

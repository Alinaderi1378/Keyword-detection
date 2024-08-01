import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# مطمئن شوید که کتابخانه‌های مورد نیاز را دانلود کرده‌اید
nltk.download('punkt')


def extract_keywords(text, top_n=10):
    # تقسیم متن به جملات و سپس به کلمات
    words = nltk.word_tokenize(text)

    # حذف کلمات غیر ضروری (مانند حروف اضافه)
    stopwords = set(nltk.corpus.stopwords.words('persian'))
    filtered_words = [word for word in words if word not in stopwords and len(word) > 1]

    # تبدیل لیست کلمات به متن
    filtered_text = ' '.join(filtered_words)

    # استفاده از TF-IDF برای استخراج کلمات کلیدی
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([filtered_text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = tfidf_matrix.toarray().flatten().argsort()[::-1]

    # استخراج بهترین کلمات کلیدی
    top_keywords = feature_array[tfidf_sorting][:top_n]

    return top_keywords

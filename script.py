import pandas as pd
import nltk
import re
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.stem.snowball import GermanStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import gensim
from gensim import corpora
from bertopic import BERTopic
from pandarallel import pandarallel

# --- NLTK Ressourcen herunterladen ---
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
    
def preprocess_text(text):
    import re
    import nltk
    from spellchecker import SpellChecker
    from nltk.stem.snowball import GermanStemmer
    from nltk.corpus import stopwords
    
    spell = SpellChecker(language='de')
    stemmer = GermanStemmer()
    stop_words = set(stopwords.words('german'))
    
    # Noise Removal (Sonderzeichen, Emojis, Zahlen entfernen)
    text = re.sub(r'[^a-zA-ZäöüÄÖÜß\s]', '', text)
    
    # Tokenisierung
    words = nltk.word_tokenize(text, language='german')
    
    clean_words = []
    
    for word in words:
        # Rechtschreibkorrektur
        corrected_word = spell.correction(word)
        if corrected_word is None: 
            corrected_word = word # Fallback, falls Korrektur fehlschlägt
            
        # Stop-Word Removal & Lowercasing
        if corrected_word.lower() not in stop_words:
            # Lemmatisierung
            stemmed_word = stemmer.stem(corrected_word)
            clean_words.append(stemmed_word)
            
    return clean_words


if __name__ == '__main__':
    # ==========================================
    # SCHRITT 1: DATEN LADEN
    # ==========================================

    # Parquet-Datei laden
    df = pd.read_parquet('Dataset\\edit_amazon_reviews_multi_de_train.parquet')

    # ==========================================
    # SCHRITT 2: VORVERARBEITUNG (NLTK & Spellchecker)
    # ==========================================

    print("\nStarte Vorverarbeitung...")

    # Worker-Prozesse initialisieren
    pandarallel.initialize(progress_bar=True, verbose=1)

    # Wörterliste für LDA speichern
    df['processed_tokens'] = df['review_body'].parallel_apply(preprocess_text)
    # ganze Sätze für BERTopic/SBERT speichern
    df['processed_text'] = df['processed_tokens'].parallel_apply(lambda x: " ".join(x))

    # ==========================================
    # SCHRITT 3: VEKTORISIERUNG
    # ==========================================

    print("Starte Vektorisierung...")

    # TF-IDF
    max_df = 0.95
    min_df = 1
    print("Erstelle Document-Term-Matrix mit TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(max_df, min_df)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])

    # SBERT
    print("Erstelle Embeddings mit SBERT...")
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = embedding_model.encode(df['review_body'].tolist(), show_progress_bar=True)

    # ==========================================
    # SCHRITT 4: THEMENEXTRAKTION
    # ==========================================

    # --- Methode A: LDA (Gensim) ---
    print("\n--- Start LDA ---")
    # Wörterbuch erstellen
    dictionary = corpora.Dictionary(df['processed_tokens'])
    # Dokument-Term-Matrix erstellen
    corpus = [dictionary.doc2bow(text) for text in df['processed_tokens']]

    # LDA Modell trainieren
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=dictionary,
                                               num_topics=5,
                                               random_state=42,
                                               passes=10)

    # --- Methode B: BERTopic ---
    print("\n--- Start BERTopic ---")
    topic_model = BERTopic(embedding_model=embedding_model, language="german", verbose=True)

    # Training (Fit)
    topics, probs = topic_model.fit_transform(df['review_body'].tolist(), embeddings)

    # ==========================================
    # SCHRITT 5: Ergebnisse anzeigen
    # ==========================================

    # LDA
    print("Gefundene Themen (LDA):")
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic: {idx} \nWords: {topic}")

    # BERTopic
    print("Gefundene Themen (BERTopic):")
    freq = topic_model.get_topic_info()
    print(freq.head())
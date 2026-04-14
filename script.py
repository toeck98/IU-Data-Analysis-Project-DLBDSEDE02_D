import pandas as pd
import nltk
import re
import argparse
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.stem.snowball import GermanStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
from bertopic import BERTopic
from pandarallel import pandarallel

# --- NLTK Ressourcen herunterladen ---
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


# --- Globale Stopwords-Liste erstellen ---
GERMAN_STOPWORDS = set(stopwords.words('german'))
GERMAN_STOPWORDS.update(["ich", "mich", "mir", "mein", "meine", "wir", "uns", "man", "es",
    "habe", "hat", "hatte", "hätte", "bin", "ist", "war", "wäre", "wurde", "wurden",
    "geht", "ging", "gibt", "kam", "kommen", "machen", "macht", "getan",
    "da", "dann", "also", "aber", "oder", "und", "für", "auf", "in", 
    "beim", "dass", "das", "die", "der", "den", "dem", "des", "ein", "eine", 
    "einen", "einem", "einer", "nicht", "nur", "auch", "so", "wie", "als"])


# --- Hilfsfunktionen für Speichern/Laden ---
def get_filename(base_name, output_dir, add_timestamp=False, extension='pkl'):
    if add_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{base_name}_{timestamp}.{extension}"
    else:
        filename = f"{base_name}.{extension}"
    return Path(output_dir) / filename


def save_step(step_num, data, output_dir, add_timestamp=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if step_num == 2:
        # Vorverarbeitete Daten speichern 
        filepath = get_filename('step2_data', output_dir, add_timestamp, 'parquet')
        data['df'].to_parquet(filepath)
        print(f"✓ Schritt 2 gespeichert: {filepath}")
    
    elif step_num == 3:
        # Vektorisierte Daten speichern
        filepath = get_filename('step3_data', output_dir, add_timestamp, 'joblib')
        joblib.dump({
            'df': data['df'],
            'tfidf_vectorizer': data['tfidf_vectorizer'],
            'tfidf_matrix': data['tfidf_matrix'],
            'embeddings': data['embeddings'],
            'embedding_model_name': data['embedding_model_name']
        }, filepath, compress=3)
        print(f"✓ Schritt 3 gespeichert: {filepath}")
    
    elif step_num == 4:
        # Trainierte Themenmodelle speichern
        filepath = get_filename('step4_data', output_dir, add_timestamp, 'joblib')
        
        joblib.dump({
            'df': data['df'],
            'dictionary': data['dictionary'],
            'corpus': data['corpus'],
            'topics': data['topics'],
            'probs': data['probs'],
            'lda_model': data['lda_model'],
            'lsa_model': data['lsa_model'],
            'tfidf_vectorizer': data['tfidf_vectorizer'],
            'topic_model': data['topic_model'],
            'lda_coherence': data.get('lda_coherence'),
            'lsa_coherence': data.get('lsa_coherence'),
            'bertopic_coherence': data.get('bertopic_coherence')
        }, filepath, compress=3)
        print(f"✓ Schritt 4 gespeichert: {filepath}")
    
    elif step_num == 5:
        # Ergebnisse als Text speichern
        filepath = get_filename('step5_results', output_dir, add_timestamp, 'txt')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ERGEBNISSE DER THEMENEXTRAKTION\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("LDA Themen:\n")
            f.write("-" * 60 + "\n")
            if data.get('lda_coherence') is not None:
                f.write(f"Coherence: {data['lda_coherence']:.4f}\n\n")
            else:
                f.write("Coherence: nicht verfügbar\n\n")
            for idx, topic in data['lda_topics']:
                f.write(f"Topic {idx}:\n{topic}\n\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("LSA Themen:\n")
            f.write("-" * 60 + "\n")
            if data.get('lsa_coherence') is not None:
                f.write(f"Coherence: {data['lsa_coherence']:.4f}\n\n")
            else:
                f.write("Coherence: nicht verfügbar\n\n")
            for idx, topic in data['lsa_topics']:
                f.write(f"Topic {idx}:\n{topic}\n\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("BERTopic Themen:\n")
            f.write("-" * 60 + "\n")
            if data.get('bertopic_coherence') is not None:
                f.write(f"Coherence: {data['bertopic_coherence']:.4f}\n\n")
            else:
                f.write("Coherence: nicht verfügbar\n\n")
            f.write(data['bertopic_info'].to_string())
        
        print(f"✓ Schritt 5 gespeichert: {filepath}")


def load_step(step_num, output_dir, add_timestamp=False):
    output_dir = Path(output_dir)
    
    # Bei Zeitstempeln die neueste Datei finden
    if add_timestamp:
        extension = 'parquet' if step_num in [1, 2] else 'joblib'
        pattern = f"step{step_num}_data*.{extension}"
        files = list(output_dir.glob(pattern))
        if not files:
            return None
        # Sortiere nach Änderungszeit und nimm die neueste
        filepath = max(files, key=lambda p: p.stat().st_mtime)
    else:
        extension = 'parquet' if step_num in [1, 2] else 'joblib'
        filepath = get_filename(f"step{step_num}_data", output_dir, False, extension)
    
    data = {}
    
    try:
        if step_num in [1, 2]:
            data['df'] = pd.read_parquet(filepath)
        
        elif step_num in [3, 4]:
            data = joblib.load(filepath)        
        
        print(f"✓ Schritt {step_num} geladen: {filepath}")
        return data
    
    except FileNotFoundError as e:
        print(f"✗ Fehler beim Laden von Schritt {step_num}: Datei nicht gefunden")
        print(f"  {e}")
        return None
    except Exception as e:
        print(f"✗ Fehler beim Laden von Schritt {step_num}: {e}")
        return None

    
def preprocess_text(text):
    import re
    import nltk
    from spellchecker import SpellChecker
    from nltk.stem.snowball import GermanStemmer
    
    spell = SpellChecker(language='de')
    stemmer = GermanStemmer()
    stop_words = GERMAN_STOPWORDS
    
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


def get_lsa_topics(lsa_model, feature_names, top_n=10):
    lsa_topics = []
    for idx, component in enumerate(lsa_model.components_):
        top_term_indices = component.argsort()[-top_n:][::-1]
        terms_with_weights = [f"{component[i]:.3f}*\"{feature_names[i]}\"" for i in top_term_indices]
        lsa_topics.append((idx, " + ".join(terms_with_weights)))
    return lsa_topics


def get_lsa_topic_words(lsa_model, feature_names, top_n=10):
    topic_words = []
    for component in lsa_model.components_:
        top_term_indices = component.argsort()[-top_n:][::-1]
        topic_words.append([feature_names[i] for i in top_term_indices])
    return topic_words


def calculate_coherence_score(topics_words, texts, dictionary, coherence='c_v'):
    if not topics_words:
        return None
    coherence_model = CoherenceModel(
        topics=topics_words,
        texts=texts,
        dictionary=dictionary,
        coherence=coherence
    )
    return float(coherence_model.get_coherence())


def get_bertopic_topic_words(topic_model, top_n=10):
    topics_dict = topic_model.get_topics()
    topic_words = []
    for topic_id, word_tuples in topics_dict.items():
        if topic_id == -1:
            continue  # -1 ist Outlier/Noise-Cluster
        words = [word for word, _ in word_tuples[:top_n]]
        if words:
            topic_words.append(words)
    return topic_words


if __name__ == '__main__':
    # ==========================================
    # Variablen initialisieren
    # ==========================================
    
    df = None
    tfidf_vectorizer = None
    tfidf_matrix = None
    embedding_model = None
    embeddings = None
    dictionary = None
    corpus = None
    lda_model = None
    lsa_model = None
    topic_model = None
    topics = None
    probs = None
    lda_coherence = None
    lsa_coherence = None
    bertopic_coherence = None

    # ==========================================
    # PARAMETER VERARBEITUNG
    # ==========================================
    
    parser = argparse.ArgumentParser(
        description='Datenanalyse-Skript mit Zwischenspeicherung',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Beispiele:
              python script.py --output-dir ./output
              python script.py --output-dir ./output --start-step 3
              python script.py --output-dir ./output --add-timestamp
              python script.py --output-dir ./output --start-step 2 --add-timestamp
        """
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Ordner für Zwischenspeicherdateien (erforderlich)'
    )
    
    parser.add_argument(
        '--start-step',
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help='Schritt, bei dem begonnen werden soll (1-5, Standard: 1)'
    )
    
    parser.add_argument(
        '--add-timestamp',
        action='store_true',
        help='Zeitstempel an Dateinamen anhängen (Standard: nein)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("KONFIGURATION")
    print("=" * 60)

    # Ordner überprüfen oder abfragen
    if not args.output_dir:
        output_dir = input("Bitte geben Sie den Ordner für Zwischenspeicherdateien ein: ").strip()
        if not output_dir:
            print("✗ Fehler: Kein Ordner angegeben. Programmabbruch.")
            exit(1)
    else:
        output_dir = args.output_dir
    
    # Entferne Anführungszeichen falls vorhanden
    output_dir = output_dir.strip('\'"')
    
    start_step = args.start_step
    add_timestamp = args.add_timestamp
    
    print(f"Output-Verzeichnis: {output_dir}")
    print(f"Start-Schritt: {start_step}")
    print(f"Zeitstempel hinzufügen: {'Ja' if add_timestamp else 'Nein'}")
    print("=" * 60 + "\n")

    # ==========================================
    # SCHRITT 1: DATEN LADEN
    # ==========================================
    
    if start_step in [1,2]:
        print("\n" + "=" * 60)
        print("SCHRITT 1: DATEN LADEN")
        print("=" * 60)
        
        # Rohdaten aus Parquet-Datei laden (Amazon Bewertungen auf Deutsch)
        df = pd.read_parquet('Dataset\\edit_amazon_reviews_multi_de_train.parquet')
        print(f"✓ Rohdaten geladen: {len(df)} Bewertungen gelesen")
        print(f"  Spalten: {', '.join(df.columns[:5])}...")
    
    if start_step > 2:
        # Lade vorherigen Schritt
        if start_step == 2:
            print("\nÜberspringe Schritt 1, lade Daten von Schritt 1...")
        else:
            print(f"\nÜberspringe Schritte 1-{start_step-1}, lade Daten von Schritt {start_step-1}...")
        loaded_data = load_step(start_step - 1, output_dir, add_timestamp)
        
        if loaded_data is None:
            print(f"✗ WARNUNG: Datei von Schritt {start_step-1} nicht gefunden in '{output_dir}'")
            print(f"  → Mögliche Ursachen: 1) Ordner existiert nicht, 2) andere --add-timestamp Einstellung")
            print(f"  → Starte neu bei Schritt 1...\n")
            start_step = 1
            df = pd.read_parquet('Dataset\\edit_amazon_reviews_multi_de_train.parquet')
            save_step(1, {'df': df}, output_dir, add_timestamp)
        else:
            df = loaded_data['df']
            
            # Weitere Daten je nach geladenem Schritt
            if start_step > 3:
                tfidf_vectorizer = loaded_data.get('tfidf_vectorizer')
                tfidf_matrix = loaded_data.get('tfidf_matrix')
                embeddings = loaded_data.get('embeddings')
                embedding_model_name = loaded_data.get('embedding_model_name')
                # Embedding Model neu laden
                if embedding_model_name and start_step == 4:
                    embedding_model = SentenceTransformer(embedding_model_name)
            
            if start_step > 4:
                dictionary = loaded_data.get('dictionary')
                corpus = loaded_data.get('corpus')
                lda_model = loaded_data.get('lda_model')
                lsa_model = loaded_data.get('lsa_model')
                topic_model = loaded_data.get('topic_model')
                topics = loaded_data.get('topics')
                probs = loaded_data.get('probs')
                lda_coherence = loaded_data.get('lda_coherence')
                lsa_coherence = loaded_data.get('lsa_coherence')
                bertopic_coherence = loaded_data.get('bertopic_coherence')

    # ==========================================
    # SCHRITT 2: VORVERARBEITUNG (NLTK & Spellchecker)
    # ==========================================
    
    if start_step <= 2:
        print("\n" + "=" * 60)
        print("SCHRITT 2: VORVERARBEITUNG (NLTK & SPELLCHECKER)")
        print("=" * 60)
        
        print("\nStarte Vorverarbeitung...")

        # Worker-Prozesse initialisieren
        pandarallel.initialize(progress_bar=True, verbose=1)

        # 1. Tokenisiere & bereinige Texte → Wörterliste (für LDA)
        df['processed_tokens'] = df['review_body'].parallel_apply(preprocess_text)
        # 2. kombiniere Tokens → String (für TF-IDF)
        df['processed_text'] = df['processed_tokens'].parallel_apply(lambda x: " ".join(x))
        
        print(f"✓ Vorverarbeitung abgeschlossen")
        
        # Zwischenspeichern in Datei
        save_step(2, {'df': df}, output_dir, add_timestamp)

    # ==========================================
    # SCHRITT 3: VEKTORISIERUNG
    # ==========================================
    
    if start_step <= 3:
        print("\n" + "=" * 60)
        print("SCHRITT 3: VEKTORISIERUNG")
        print("=" * 60)
        
        print("\nStarte Vektorisierung (TF-IDF + Deep Learning Embeddings)...")

        # === TF-IDF: Traditional Bag-of-Words Vektorisierung (für LDA) ===
        # Parameter: Filtere zu häufige Wörter (>85% Docs) und zu seltene (<0.3% Docs)
        max_document_frequency = 0.85
        min_document_frequency = 0.003
        
        print(f"\n  1. TF-IDF Vektorisierung (max_df={max_document_frequency}, min_df={min_document_frequency})...")
        tfidf_vectorizer = TfidfVectorizer(max_df=max_document_frequency, min_df=min_document_frequency)
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])
        n_features = tfidf_matrix.shape[1]
        sparsity = 100 * (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))
        print(f"    ✓ TF-IDF Matrix: {tfidf_matrix.shape[0]} Dokumente × {n_features} Features")
        print(f"    Sparsität: {sparsity:.1f}% (dünn besetzt = gut)")

        # === SBERT: Deep Learning Embeddings (für BERTopic) ===
        # Nutze ORIGINAL-Texte (nicht processed) um semantischen Kontext zu bewahren
        print(f"\n  2. SBERT Deep Learning Embeddings (multilingual-MiniLM-L12)...")
        embedding_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        embedding_model = SentenceTransformer(embedding_model_name)
        embeddings = embedding_model.encode(df['review_body'].tolist(), show_progress_bar=True)
        print(f"    ✓ Embeddings erstellt: {embeddings.shape[0]} Dokumente × {embeddings.shape[1]} Dimensionen")
        
        # Zwischenspeichern in Datei
        save_step(3, {
            'df': df,
            'tfidf_vectorizer': tfidf_vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'embeddings': embeddings,
            'embedding_model_name': embedding_model_name
        }, output_dir, add_timestamp)

    # ==========================================
    # SCHRITT 4: THEMENEXTRAKTION
    # ==========================================
    
    if start_step <= 4:
        print("\n" + "=" * 60)
        print("SCHRITT 4: THEMENEXTRAKTION")
        print("=" * 60)

        # === METHODE A: LDA (Latent Dirichlet Allocation) ===
        print("\n  3. LDA Topic-Modellierung...")
        
        # Erstelle Gensim-Wörterbuch
        dictionary = corpora.Dictionary(df['processed_tokens'])
        print(f"    Vokabular: {len(dictionary)} unique Tokens")
        
        # Konvertiere zu Bow-Format
        corpus = [dictionary.doc2bow(text) for text in df['processed_tokens']]
        
        # Trainiere LDA Modell
        print(f"    Trainiere LDA...")
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                   id2word=dictionary,
                                                   num_topics=12,
                                                   random_state=42,
                                                   passes=10)
        print(f"    ✓ LDA Modell trainiert und konvergiert")

        lda_topic_words = [
            [word for word, _ in topic_terms]
            for _, topic_terms in lda_model.show_topics(num_topics=-1, num_words=10, formatted=False)
        ]
        lda_coherence = calculate_coherence_score(
            topics_words=lda_topic_words,
            texts=df['processed_tokens'].tolist(),
            dictionary=dictionary,
            coherence='c_v'
        )
        print(f"    Coherence Score: {lda_coherence:.4f}")

        # === METHODE B: LSA (Latent Semantic Analysis) ===
        print("\n  4. LSA Topic-Modellierung...")
        n_topics = 12
        print(f"    Trainiere LSA...")
        lsa_model = TruncatedSVD(n_components=n_topics, random_state=42)
        lsa_model.fit(tfidf_matrix)
        explained_variance = float(np.sum(lsa_model.explained_variance_ratio_))
        print(f"    ✓ LSA Modell trainiert: {n_topics} Komponenten, erklärte Varianz={explained_variance:.2%}")

        lsa_topic_words = get_lsa_topic_words(lsa_model, tfidf_vectorizer.get_feature_names_out(), top_n=10)
        lsa_coherence = calculate_coherence_score(
            topics_words=lsa_topic_words,
            texts=df['processed_tokens'].tolist(),
            dictionary=dictionary,
            coherence='c_v'
        )
        print(f"    Coherence Score: {lsa_coherence:.4f}")

        # === METHODE C: BERTopic (Modern Deep Learning) ===
        print("\n  5. BERTopic Topic-Modellierung (moderner kontextbasierter Ansatz)...")
        
        # Embedding Model laden falls noch nicht geladen
        if embedding_model is None:
            embedding_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
            embedding_model = SentenceTransformer(embedding_model_name)
        
        # Konfiguriere BERTopic mit Stopword-Filterung & Hyperparametern
        vectorizer = CountVectorizer(stop_words=list(GERMAN_STOPWORDS),  # Filtere Füllwörter
                                     max_features=1000,  # Begrenze Features auf Top 1000
                                     lowercase=True, 
                                     ngram_range=(1, 2))  # Einzelne Wörter + Bigrams
        
        topic_model = BERTopic(embedding_model=embedding_model, 
                              language="german", 
                              verbose=False,
                              nr_topics=12,
                              min_topic_size=50,
                              vectorizer_model=vectorizer)

        # Training
        print(f"    Trainiere BERTopic mit vorgenerierten Embeddings...")
        topics, probs = topic_model.fit_transform(df['review_body'].tolist(), embeddings)
        n_topics_found = len(set(topics)) - (1 if -1 in set(topics) else 0)  # Zähle gefundene Topics (ohne Noise)
        print(f"    ✓ BERTopic Modell trainiert: {n_topics_found} Topics gefunden")

        bertopic_coherence = None
        try:
            bertopic_topic_words = get_bertopic_topic_words(topic_model, top_n=10)
            analyzer = vectorizer.build_analyzer()
            bertopic_texts = [analyzer(doc) for doc in df['review_body'].tolist()]
            bertopic_dictionary = corpora.Dictionary(bertopic_texts)
            # Für BERTopic nutzen wir denselben Tokenizer wie im BERTopic-CountVectorizer.
            bertopic_coherence = calculate_coherence_score(
                topics_words=bertopic_topic_words,
                texts=bertopic_texts,
                dictionary=bertopic_dictionary,
                coherence='c_v'
            )
            print(f"    Coherence Score: {bertopic_coherence:.4f}")
        except Exception as e:
            print(f"    Coherence Score für BERTopic nicht berechenbar: {e}")
        
        # Zwischenspeichern in Datei
        save_step(4, {
            'df': df,
            'lda_model': lda_model,
            'lsa_model': lsa_model,
            'tfidf_vectorizer': tfidf_vectorizer,
            'dictionary': dictionary,
            'corpus': corpus,
            'topic_model': topic_model,
            'topics': topics,
            'probs': probs,
            'lda_coherence': lda_coherence,
            'lsa_coherence': lsa_coherence,
            'bertopic_coherence': bertopic_coherence
        }, output_dir, add_timestamp)

    # ==========================================
    # SCHRITT 5: Ergebnisse anzeigen
    # ==========================================
    
    if start_step <= 5:
        print("\n" + "=" * 60)
        print("SCHRITT 5: ERGEBNISSE ANZEIGEN")
        print("=" * 60)

        # === LDA Topic-Ergebnisse ===
        print("\n" + "=" * 60)
        print("ERGEBNISSE: LDA TOPICS (Top 10 Wörter pro Topic)")
        print("=" * 60)
        print("(Format: Gewicht * Wort + Gewicht * Wort...)\n")
        if lda_coherence is not None:
            print(f"Coherence Score: {lda_coherence:.4f}\n")
        else:
            print("Coherence Score: nicht verfügbar\n")

        lda_topics = []
        for idx, topic in lda_model.print_topics(-1):
            print(f"Topic {idx}:")
            print(f"  {topic}\n")
            lda_topics.append((idx, topic))

        # === LSA Topic-Ergebnisse ===
        print("\n" + "=" * 60)
        print("ERGEBNISSE: LSA TOPICS (Top 10 Wörter pro Topic)")
        print("=" * 60)
        print("(Format: Gewicht * Wort + Gewicht * Wort...)\n")
        lsa_topics = []
        if lsa_model is not None and tfidf_vectorizer is not None:
            if lsa_coherence is not None:
                print(f"Coherence Score: {lsa_coherence:.4f}\n")
            else:
                print("Coherence Score: nicht verfügbar\n")

            feature_names = tfidf_vectorizer.get_feature_names_out()
            lsa_topics = get_lsa_topics(lsa_model, feature_names, top_n=10)
            for idx, topic in lsa_topics:
                print(f"Topic {idx}:")
                print(f"  {topic}\n")
            lsa_explained_variance = float(np.sum(lsa_model.explained_variance_ratio_))
            print(f"Erklärte Varianz (kumuliert): {lsa_explained_variance:.2%}")
        else:
            lsa_explained_variance = None
            print("LSA Ergebnisse nicht verfügbar (fehlendes Modell oder fehlender TF-IDF-Vektorisierer).")

        # === BERTopic Topic-Ergebnisse ===
        print("\n" + "=" * 60)
        print("ERGEBNISSE: BERTOPIC TOPICS (Semantisch gruppiert)")
        print("=" * 60)
        print("\nTop 10 Topics nach Dokumentanzahl:\n")

        if bertopic_coherence is not None:
            print(f"Coherence Score: {bertopic_coherence:.4f}\n")
        else:
            print("Coherence Score: nicht verfügbar\n")

        freq = topic_model.get_topic_info()
        # Zeige erweiterte Statistiken
        print(freq[['Topic', 'Count', 'Name']].head(10).to_string(index=False))
        print(f"\n... ({len(freq)-1} Topics insgesamt, Topic -1 = Noise/Outlier)")
        
        # Speichern in Datei
        save_step(5, {
            'lda_topics': lda_topics,
            'lsa_topics': lsa_topics,
            'lsa_explained_variance': lsa_explained_variance,
            'bertopic_info': freq,
            'lda_coherence': lda_coherence,
            'lsa_coherence': lsa_coherence,
            'bertopic_coherence': bertopic_coherence
        }, output_dir, add_timestamp)
    
    print("\n" + "=" * 60)
    print("SKRIPT ABGESCHLOSSEN")
    print("=" * 60)
    print(f"Alle Zwischenergebnisse wurden in '{output_dir}' gespeichert.")
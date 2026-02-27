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


# --- Hilfsfunktionen für Speichern/Laden ---
def get_filename(base_name, output_dir, add_timestamp=False, extension='pkl'):
    """Erstellt Dateinamen mit optionalem Zeitstempel"""
    if add_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{base_name}_{timestamp}.{extension}"
    else:
        filename = f"{base_name}.{extension}"
    return Path(output_dir) / filename


def save_step(step_num, data, output_dir, add_timestamp=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if step_num == 1:
        # DataFrame als Parquet speichern
        filepath = get_filename('step1_data', output_dir, add_timestamp, 'parquet')
        data['df'].to_parquet(filepath)
        print(f"✓ Schritt 1 gespeichert: {filepath}")
    
    elif step_num == 2:
        # DataFrame mit Vorverarbeitung als Parquet speichern
        filepath = get_filename('step2_data', output_dir, add_timestamp, 'parquet')
        data['df'].to_parquet(filepath)
        print(f"✓ Schritt 2 gespeichert: {filepath}")
    
    elif step_num == 3:
        # Alle Vektorisierungsdaten in einer joblib-Datei speichern
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
        # Alle Themenextraktionsdaten in einer joblib-Datei speichern
        filepath = get_filename('step4_data', output_dir, add_timestamp, 'joblib')
        
        joblib.dump({
            'df': data['df'],
            'dictionary': data['dictionary'],
            'corpus': data['corpus'],
            'topics': data['topics'],
            'probs': data['probs'],
            'lda_model': data['lda_model'],
            'topic_model': data['topic_model']
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
            for idx, topic in data['lda_topics']:
                f.write(f"Topic {idx}:\n{topic}\n\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("BERTopic Themen:\n")
            f.write("-" * 60 + "\n")
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
    topic_model = None
    topics = None
    probs = None

    # ==========================================
    # PARAMETER VERARBEITUNG
    # ==========================================
    
    parser = argparse.ArgumentParser(
        description='Datenanalyse-Pipeline mit Zwischenspeicherung',
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
    
    if start_step <= 1:
        print("\n" + "=" * 60)
        print("SCHRITT 1: DATEN LADEN")
        print("=" * 60)
        
        # Parquet-Datei laden
        df = pd.read_parquet('Dataset\\edit_amazon_reviews_multi_de_train.parquet')
        print(f"✓ Daten geladen: {len(df)} Zeilen")
        
        # Zwischenspeichern in Datei
        save_step(1, {'df': df}, output_dir, add_timestamp)
    
    elif start_step > 1:
        # Lade vorherigen Schritt
        if start_step == 2:
            print("\nÜberspringe Schritt 1, lade Daten von Schritt 1...")
        else:
            print(f"\nÜberspringe Schritte 1-{start_step-1}, lade Daten von Schritt {start_step-1}...")
        loaded_data = load_step(start_step - 1, output_dir, add_timestamp)
        
        if loaded_data is None:
            print(f"✗ Fehler: Kann Schritt {start_step-1} nicht laden. Starte bei Schritt 1.")
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
                topic_model = loaded_data.get('topic_model')
                topics = loaded_data.get('topics')
                probs = loaded_data.get('probs')

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

        # Wörterliste für LDA speichern
        df['processed_tokens'] = df['review_body'].parallel_apply(preprocess_text)
        # ganze Sätze für BERTopic/SBERT speichern
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
        
        print("\nStarte Vektorisierung...")

        # TF-IDF
        max_document_frequency = 0.85
        min_document_frequency = 0.003
        print("Erstelle Document-Term-Matrix mit TF-IDF...")
        tfidf_vectorizer = TfidfVectorizer(max_df=max_document_frequency, min_df=min_document_frequency)
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])
        print(f"✓ TF-IDF Matrix erstellt: {tfidf_matrix.shape}")

        # SBERT
        print("Erstelle Embeddings mit SBERT...")
        embedding_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        embedding_model = SentenceTransformer(embedding_model_name)
        embeddings = embedding_model.encode(df['review_body'].tolist(), show_progress_bar=True)
        print(f"✓ Embeddings erstellt: {embeddings.shape}")
        
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

        # --- Methode A: LDA ---
        print("\n--- Start LDA ---")
        # Wörterbuch erstellen
        dictionary = corpora.Dictionary(df['processed_tokens'])
        # Dokument-Term-Matrix erstellen
        corpus = [dictionary.doc2bow(text) for text in df['processed_tokens']]

        # LDA Modell trainieren
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                   id2word=dictionary,
                                                   num_topics=12,
                                                   random_state=42,
                                                   passes=10)
        print(f"✓ LDA Modell trainiert")

        # --- Methode B: BERTopic ---
        print("\n--- Start BERTopic ---")
        
        # Embedding Model laden falls noch nicht geladen
        if embedding_model is None:
            embedding_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
            embedding_model = SentenceTransformer(embedding_model_name)
        
        topic_model = BERTopic(embedding_model=embedding_model, language="german", verbose=True, nr_topics=12, min_topic_size=50)

        # Training (Fit)
        topics, probs = topic_model.fit_transform(df['review_body'].tolist(), embeddings)
        print(f"✓ BERTopic Modell trainiert")
        
        # Zwischenspeichern in Datei
        save_step(4, {
            'df': df,
            'lda_model': lda_model,
            'dictionary': dictionary,
            'corpus': corpus,
            'topic_model': topic_model,
            'topics': topics,
            'probs': probs
        }, output_dir, add_timestamp)

    # ==========================================
    # SCHRITT 5: Ergebnisse anzeigen
    # ==========================================
    
    if start_step <= 5:
        print("\n" + "=" * 60)
        print("SCHRITT 5: ERGEBNISSE ANZEIGEN")
        print("=" * 60)

        # LDA
        print("\nGefundene Themen (LDA):")
        print("-" * 60)
        lda_topics = []
        for idx, topic in lda_model.print_topics(-1):
            print(f"Topic: {idx} \nWords: {topic}\n")
            lda_topics.append((idx, topic))

        # BERTopic
        print("\nGefundene Themen (BERTopic):")
        print("-" * 60)
        freq = topic_model.get_topic_info()
        print(freq.head())
        
        # Zwischenspeichern in Datei
        save_step(5, {
            'lda_topics': lda_topics,
            'bertopic_info': freq
        }, output_dir, add_timestamp)
    
    print("\n" + "=" * 60)
    print("SKRIPT ABGESCHLOSSEN")
    print("=" * 60)
    print(f"Alle Zwischenergebnisse wurden in '{output_dir}' gespeichert.")
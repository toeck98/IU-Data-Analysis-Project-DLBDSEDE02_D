## Voraussetzungen
- Python 3.9+
- pip

## Basis-Kommando

```bash
python script.py --output-dir <AUSGABEVERZEICHNIS>
```

## Parameter

| Parameter | Beschreibung | Standard |
|-----------|-------------|----------|
| `--output-dir` | Pfad zum Ausgabeverzeichnis | **Pflichtfeld** |
| `--start-step` | Bei welchem Schritt starten (1-5) | 1 |
| `--add-timestamp` | Zeitstempel zu Dateinamen hinzufügen | False |

### Beispiele

**Vollständiger Durchlauf:**
```bash
python script.py --output-dir ./output
```

**Ab Schritt 3 fortsetzen:**
```bash
python script.py --output-dir ./output --start-step 3
```

**Mit Zeitstempel-Dateinamen:**
```bash
python script.py --output-dir ./output --add-timestamp
```

**Kombination:**
```bash
python script.py --output-dir ./output --start-step 2 --add-timestamp
```
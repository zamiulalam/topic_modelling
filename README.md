# BERTopic Topic Modeling on NYT Artificial Intelligence Coverage

A topic modeling pipeline applied to New York Times articles about Artificial Intelligence, using [BERTopic](https://maartengr.github.io/BERTopic/) to discover and visualize latent themes across a corpus of ~10,000 articles.

---

## Overview

This project analyzes how the New York Times has covered Artificial Intelligence over time by applying transformer-based topic modeling to article headlines and leads. The pipeline ingests structured article data exported from a news database, preprocesses the text, fits a BERTopic model, and produces interactive visualizations of the discovered topics.

The model identified **110 coherent topics** across 10,267 articles, spanning themes such as stock markets, energy policy, the Israel-Gaza conflict, education, AI regulation, OpenAI leadership, facial recognition, semiconductor chips, Hollywood labor strikes, and more.

---

## Data

- **Source:** New York Times articles retrieved via a news database (e.g., Factiva/ProQuest), queried on the keyword *Artificial Intelligence*
- **Format:** 11 `.XLSX` files (`Results list for_Artificial Intelligence 1.XLSX` through `...11.XLSX`)
- **Total records:** 10,267 articles
- **Date range:** ~2022–2025
- **Key columns used:**
  - `Hlead` — Headline + article lead paragraph (primary input to the model)
  - `Title`, `Published date`, `Section`, `Byline`, `Word count`, `Publication type`

> **Note:** The raw data files are not included in this repository due to licensing restrictions. You will need access to the original export files to reproduce results.

---

## Pipeline

### 1. Data Ingestion
Eleven Excel files are loaded individually and concatenated into a single DataFrame with `pandas`.

### 2. Text Preprocessing
A custom `preprocess_text()` function is applied to the `Hlead` column to clean the text before modeling:
- Removes em-dashes (`—`)
- Strips month abbreviations (Jan, Feb, etc.)
- Removes standalone date numbers (1–31)
- Removes parenthetical source attributions (e.g., publication names in parentheses), while preserving `(AI)`

The cleaned text is stored in a new column `Hleadz`.

### 3. Topic Modeling with BERTopic
A `BERTopic` model is configured and fit on the cleaned document list:

```python
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=CountVectorizer(stop_words="english"),
    min_topic_size=20
)

topics, probs = topic_model.fit_transform(docs)
```

### 4. Visualization
The notebook renders interactive Plotly visualizations via BERTopic's built-in methods:
- **`topic_model.get_topic_info()`** — Summary table of all topics with representative keywords and document counts
- **`topic_model.get_representative_docs(topic_id)`** — Retrieve sample documents for a specific topic
- **`topic_model.visualize_barchart()`** — Bar chart of top words per topic with c-TF-IDF scores

---

## Selected Topics Discovered

| Topic ID | Keywords | Count |
|----------|----------|-------|
| 0 | percent, 500, stocks, wall, nasdaq | 276 |
| 1 | energy, climate, gas, power, electricity | 217 |
| 2 | israel, gaza, israeli, hamas, iran | 166 |
| 3 | students, school, schools, education, college | 155 |
| 4 | tariffs, trade, trump, tariff, european | 154 |
| 5 | ukraine, russia, russian, putin, war | 147 |
| 6 | actors, strike, studios, hollywood, writers | 142 |
| 7 | altman, openai, sam, board, chief | 130 |
| 8 | images, facial, recognition, abuse, photos | 129 |
| 9 | chips, nvidia, chip, china, chinese | 119 |
| 10 | regulate, lawmakers, california, technology, rules | 117 |
| 11 | chatbot, google, search, bing, chatbots | 110 |

*Topic -1 represents outlier/unclustered documents (3,674 articles).*

---

## Requirements

```
bertopic
pandas
openpyxl
nltk
scikit-learn
sentence-transformers
plotly
```

Install all dependencies with:

```bash
pip install bertopic pandas openpyxl nltk scikit-learn sentence-transformers plotly
```

You may also need to download NLTK resources:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

---

## Usage

1. Place your 11 Excel data files in a local directory and update the file paths in the notebook:
   ```python
   df1 = pd.read_excel('path/to/Results list for_Artificial Intelligence 1.XLSX')
   ```

2. Open and run `BERTopic_NYT.ipynb` from top to bottom.

3. Inspect topic outputs with `topic_model.get_topic_info()` and explore the interactive bar chart visualization.

---

## Notes

- The `all-MiniLM-L6-v2` sentence transformer is a lightweight, fast model well-suited for large document collections. Swapping in a larger model (e.g., `all-mpnet-base-v2`) may yield richer topic representations at the cost of runtime.
- `min_topic_size=20` means a topic must contain at least 20 documents to be retained. Lowering this value will produce more granular topics.
- Documents assigned to Topic `-1` are outliers that did not fit cleanly into any cluster. These can be reduced by using `BERTopic`'s outlier reduction methods.

---

## Project Structure

```
.
├── BERTopic_NYT.ipynb       # Main analysis notebook
├── README.md                # This file
└── data/                    # (Not included) Excel source files
    ├── Results list for_Artificial Intelligence 1.XLSX
    ├── ...
    └── Results list for_Artificial Intelligence 11.XLSX
```

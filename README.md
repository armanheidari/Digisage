# 🦉 DigiSage

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Backend-Flask%20%7C%20Flask--SocketIO-green.svg)](https://flask.palletsprojects.com/)
[![NLP](https://img.shields.io/badge/Persian%20NLP-Hazm%20%7C%20Transformers-orange.svg)](https://github.com/roopeshvs/hazm)
[![MLflow](https://img.shields.io/badge/Experiment%20Tracking-MLflow-0194E2.svg)](https://mlflow.org/)
[![LLM](https://img.shields.io/badge/LLM-Llama--3.3--70B%20(Together%20AI)-purple.svg)](https://www.together.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**DigiSage** is an intelligent, Persian-language Retrieval-Augmented Generation (RAG) assistant and semantic FAQ chatbot built specifically for [Digikala](https://www.digikala.com/). It combines specialized Persian text preprocessing, hybrid vectorization (TF-IDF and multilingual dense embeddings like LaBSE), K-Nearest Neighbors semantic ranking, automated data augmentation, and large language model generation (Llama 3.3 70B via Together AI) to deliver fast, accurate, and context-grounded customer service responses.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture & Design Patterns](#-architecture--design-patterns)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Environment Configuration](#-environment-configuration)
- [Running the Application](#-running-the-application)
- [Pipeline & Development Usage](#-pipeline--development-usage)
  - [Training Mode](#training-mode)
  - [Inference Mode](#inference-mode)
  - [Data Augmentation Pipeline](#data-augmentation-pipeline)
  - [Latency Benchmarking](#latency-benchmarking)
- [API & WebSocket Protocol](#-api--websocket-protocol)
- [Evaluation Metrics](#-evaluation-metrics)
- [License](#-license)

---

## 🌟 Overview

Customer queries in e-commerce FAQ sections frequently use colloquial Persian, informal grammar, spelling variations, or phrasing that does not strictly match canonical documentation. 

DigiSage resolves this by:
1. **Preprocessing & Normalizing** Persian text using Hazm (correcting half-spaces/ZWNJ, diacritics, numerals, informal affixes, and stop words).
2. **Indexing & Vectorizing** the corpus using statistical TF-IDF or dense sentence transformer representations (e.g. `sentence-transformers/LaBSE`).
3. **Retrieving Relevant FAQs** using Nearest Neighbor similarity search (Cosine or Jaccard).
4. **Augmenting and Grounding Responses** via Together AI's Llama 3.3 70B Instruct model, delivering fluent and polite answers in Persian directly grounded in Digikala's verified FAQ answers.
5. **Streaming Results in Real Time** via WebSockets to a dedicated interactive web UI.

---

## ✨ Key Features

- **Persian Text Processing Pipeline**: Standardizes Persian orthography, handles informal colloquial speech, performs lemmatization and stemming, and strips custom Persian stopwords.
- **Flexible Vectorization Strategies**:
  - **TF-IDF**: Configurable n-gram range, max features, and document frequencies.
  - **Dense Embeddings**: Pre-trained sentence transformers (`sentence-transformers/LaBSE`, `all-MiniLM-L6-v2`, etc.).
- **Semantic Retrieval**: K-Nearest Neighbors search with Cosine or Jaccard similarity and distance thresholding.
- **RAG Generation**: Leverages Meta's `Llama-3.3-70B-Instruct-Turbo` through Together AI to synthesize natural responses from retrieved FAQs.
- **Comprehensive Data Augmentation**:
  - T5-based paraphrasing (`humarin/chatgpt_paraphraser_on_T5_base`).
  - Round-trip translation (Persian → English → Paraphrase → Persian).
  - Synonym replacement using a curated Persian lexicon (`data/synonyms.json`).
- **Experiment Tracking**: Integrated MLflow tracking for logging hyperparameters, classification metrics, confusion matrices, and serialized model artifacts.
- **Real-Time Interactive UI**: Built with Flask-SocketIO and responsive CSS, featuring Persian typography (IRANYekan), question inspection drawers, and dual retrieval/generation views.

---

## 🏗 Architecture & Design Patterns

DigiSage is engineered using clean object-oriented design patterns:

- **Builder Pattern (`PipelineBuilder`)**: Fluent interface for assembling preprocessors, vectorizers, vocabularies, similarity metrics, evaluators, and loggers.
- **Factory & Registry Pattern**: Decoupled creation of pipeline stages (`PreprocessorFactory`, `VectorizerFactory`, `SimilaritySearchFactory`, `PipelineFactory`, etc.).
- **Strategy Pattern**: Interchangeable vectorization (`TfidfVectorizerStrategy`, `SentenceTransformerStrategy`) and similarity algorithms (`KNNSimilarityStrategy`).
- **Observer / Decorator Pattern (`PipelineLogger.observe`)**: Non-intrusive runtime tracking and MLflow parameter/metric logging.

```
User Query ──► Flask Endpoint (/submit)
                    │
                    ▼
           DigiSage Inference Pipeline
                    │
       ┌────────────┴────────────┐
       ▼                         ▼
 Hazm Preprocessing     Vectorization (TF-IDF / LaBSE)
                                 │
                                 ▼
                     K-NN Similarity Search
                                 │
                                 ▼
                    Top-K Retrieved FAQ Pairs
                                 │
       ┌─────────────────────────┴────────────────────────┐
       ▼                                                  ▼
 WebSocket Emit ('ir_results')               Together AI (Llama 3.3 70B)
       │                                                  │
       ▼                                                  ▼
  Interactive UI                               WebSocket Emit ('llm_response')
  Inspection Card                                         │
                                                          ▼
                                                    Chat Stream
```

---

## 📁 Project Structure

```text
DigiSage/
├── app.py                          # Main Flask & Socket.IO server (training + inference + UI)
├── requirements.txt                # Python project dependencies
├── .env.example                    # Environment variable configuration template
├── .env                            # Local environment secrets (API keys & tokens)
├── .gitignore                      # Git ignored files and directories
├── .project_root                   # Root marker for dynamic path resolution
├── LICENSE                         # Project license
├── README.md                       # Comprehensive documentation
│
├── path_handler/                   # Dynamic cross-platform path resolver
│   ├── __init__.py
│   └── path_manager.py
│
├── data/                           # Data directory
│   ├── digikala_faq.csv            # Original crawled Digikala FAQ dataset
│   ├── preprocessed_digikala_faq.csv # Preprocessed baseline FAQ dataset
│   ├── augmented_dataset.csv       # Augmented FAQ dataset (paraphrases + synonyms)
│   ├── test_digikala_faq.csv       # Evaluation test split
│   ├── stops.txt                   # Persian stopwords list
│   ├── synonyms.json               # Persian synonyms dictionary
│   └── vocabulary.txt              # Extracted vocabulary
│
├── src/                            # Core application source code
│   ├── generate/                   # Synthetic data generation & augmentation
│   │   ├── augmenter/              # Augmenter components (paraphraser, replacer, translator)
│   │   │   ├── base.py             # BaseAugmenter coordinating augmentation steps
│   │   │   ├── factory.py          # AugmenterFactory
│   │   │   ├── paraphraser.py      # T5-based text paraphraser
│   │   │   ├── replacer.py         # Dictionary-based synonym replacer
│   │   │   └── translator.py       # Google Translate wrapper
│   │   └── config/                 # Augmentation configuration dataclasses
│   │       ├── config.py
│   │       └── default.py
│   │
│   └── pipelines/                  # Modular retrieval & evaluation pipeline
│       ├── config/                 # Pipeline configuration dataclasses
│       │   ├── config.py
│       │   └── default.py          # Default configurations (TF-IDF, Cosine, etc.)
│       ├── evaluator/              # Model evaluation metrics
│       │   ├── analysis.py         # Metric calculation helpers
│       │   ├── base.py             # BaseEvaluator (MRR, Precision@k, Recall@k)
│       │   └── factory.py          # EvaluatorFactory
│       ├── logger/                 # Experiment tracking
│       │   ├── base.py             # Logger decorators
│       │   └── log.py              # MLflow integration and artifact serialization
│       ├── pipeline/               # Core pipeline orchestration
│       │   ├── base.py             # BasePipeline, TrainingPipeline, InferencePipeline
│       │   ├── builder.py          # PipelineBuilder
│       │   ├── factory.py          # PipelineFactory
│       │   └── registry.py         # PipelineRegistry
│       ├── preprocessor/           # Persian NLP preprocessing
│       │   ├── base.py             # BasePreprocessor
│       │   ├── components.py       # Normalizer, Stemmer, Lemmatizer, Tokenizer
│       │   ├── factory.py          # PreprocessorFactory
│       │   └── registry.py         # PreprocessorRegistry
│       ├── similarity/             # Vector similarity search
│       │   ├── base.py             # BaseSimilaritySearch
│       │   ├── factory.py          # SimilaritySearchFactory
│       │   ├── registry.py         # SimilaritySearchRegistry
│       │   └── strategy.py         # KNNSimilarityStrategy
│       ├── vectorizer/             # Vectorization strategies
│       │   ├── base.py             # BaseVectorizer
│       │   ├── factory.py          # VectorizerFactory
│       │   ├── registry.py         # VectorizerRegistry
│       │   └── strategy.py         # Tfidf, Count, SentenceTransformer strategies
│       ├── vocabulary/             # Vocabulary extraction & building
│       │   ├── base.py             # VocabularyBuilder
│       │   └── factory.py          # VocabularyBuilderFactory
│       └── type_hint.py            # Enums (ControllerType: TRAINING, INFERENCE)
│
├── dev/                            # Development, research, and benchmarking
│   ├── data_analysis.ipynb         # Exploratory data analysis notebook
│   ├── test.ipynb                  # Component testing notebook
│   ├── time.py                     # Retrieval latency benchmark script
│   └── time_experiment.png         # Benchmark latency plot
│
└── ui/                             # Frontend user interface
    ├── index.html                  # Single-page web chat application
    ├── styles/                     # CSS styling and fonts
    │   ├── style.css
    │   └── fonts/                  # IRANYekan Persian font family
    ├── js/                         # Frontend client scripts
    │   └── index.js                # WebSocket handlers & DOM interactions
    └── images/                     # Graphic assets
        └── send.png
```

---

## 🚀 Installation & Setup

### Prerequisites

- **Python**: Version 3.10 or higher.
- **Git**: Installed and available in your terminal.
- **Hugging Face Account**: (Optional, for downloading private/gated transformer models).
- **Together AI Account**: (For LLM response generation via Llama 3.3).

### 1. Clone the Repository

```bash
git clone https://github.com/armanheidari/DigiSage.git
cd DigiSage
```

### 2. Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ Environment Configuration

Copy the sample environment file to `.env`:

```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:

```ini
# Hugging Face token (for downloading transformer models)
TOKEN="your_huggingface_token_here"

# Together AI API Key (for Llama 3.3 70B response generation)
APIKEY="your_together_ai_api_key_here"
```

---

## 🖥 Running the Application

### Option A: Standard Web Application

Start the Flask server with real-time SocketIO:

```bash
python app.py
```

When started, `app.py` will:
1. Initialize and train the retrieval index using the default configuration (`PIPELINE_DEFAULT_CONFIG`).
2. Build the inference pipeline and connect to the Together AI client.
3. Serve the web application at **`http://localhost:5000`**.

Open your browser and navigate to `http://localhost:5000` to interact with **DigiSage**.

### Option B: With MLflow Experiment Tracking (Optional)

If you want to track training runs, parameters, and evaluation metrics:

1. Launch an MLflow tracking server on port `9437`:
   ```bash
   mlflow ui --port 9437
   ```
2. In a separate terminal, run `app.py` or your pipeline script:
   ```bash
   python app.py
   ```
3. Open `http://localhost:9437` to inspect parameters, artifacts, and evaluation metrics.

---

## 🔧 Pipeline & Development Usage

### Training Mode

To train the pipeline and evaluate it on the test split:

```python
from src.pipelines.type_hint import ControllerType
from src.pipelines.pipeline.builder import PipelineBuilder
from src.pipelines.config.default import PIPELINE_DEFAULT_CONFIG

# Build and execute the training pipeline
pipeline = (
    PipelineBuilder(controller_type=ControllerType.TRAINING)
    .with_logger(run_name="digisage", experiment_name="DigiSage", mode="development")
    .with_config(PIPELINE_DEFAULT_CONFIG)
    .with_preprocessor(logger=True)
    .with_vectorizer(logger=True)
    .with_vocabulary()
    .with_similarity(logger=True)
    .with_evaluator(logger=True)
    .build(logger=True)
)

vectorized_corpus = pipeline.run()
```

### Inference Mode

To query an already trained pipeline:

```python
from src.pipelines.type_hint import ControllerType
from src.pipelines.pipeline.builder import PipelineBuilder
from src.pipelines.config.default import PIPELINE_DEFAULT_CONFIG

# Build inference pipeline
inference_pipeline = (
    PipelineBuilder(controller_type=ControllerType.INFERENCE)
    .with_logger(run_name="digisage", experiment_name="DigiSage", mode="development")
    .with_config(PIPELINE_DEFAULT_CONFIG)
    .with_preprocessor()
    .with_vectorizer()
    .with_vocabulary()
    .with_similarity()
    .with_evaluator()
    .build()
)

# Run a sample query in Persian
results = inference_pipeline.run("چگونه می‌توانم سفارش خود را مرجوع کنم؟")
print("Retrieved Questions:", results["retrieved_question"])
print("Retrieved Answers:", results["retrieved_answer"])
```

### Data Augmentation Pipeline

To expand the dataset with paraphrased queries and synonyms:

```python
import asyncio
import pandas as pd
from src.generate.augmenter.factory import AugmenterFactory
from src.generate.config.default import AUGMENTER_DEFAULT_CONFIG

augmenter = AugmenterFactory.create(AUGMENTER_DEFAULT_CONFIG)
dataset = pd.read_csv("data/digikala_faq.csv")

# Run async augmentation (T5 Paraphraser + Back-translation + Synonym Replacer)
augmented_df = asyncio.run(augmenter.augment(dataset, save=True))
print(f"Augmented dataset size: {len(augmented_df)}")
```

### Latency Benchmarking

To benchmark retrieval speed across varying query lengths:

```bash
python dev/time.py
```

This will run multiple query iterations and output `time_experiment.png` depicting average retrieval time vs query length.

---

## 📡 API & WebSocket Protocol

### HTTP Endpoint

- **POST `/submit`**:
  - Request body:
    ```json
    {
      "query": "هزینه ارسال چقدر است؟"
    }
    ```
  - Response (202 Accepted):
    ```json
    {
      "status": "Processing started"
    }
    ```

### WebSocket Events

- **Client → Server**:
  - `connect`: Establishes socket connection.
- **Server → Client**:
  - `ir_results`: Emits top-$k$ retrieved FAQs (question, answer, category).
    ```json
    {
      "ir_results": [
        {
          "question": "چگونه می‌توانم هزینه ارسال را مشاهده کنم؟",
          "answer": "هزینه ارسال در مرحله نهایی ثبت سفارش با توجه به آدرس شما محاسبه می‌شود.",
          "category": "ارسال و تحویل"
        }
      ]
    }
    ```
  - `llm_response`: Emits generated conversational Persian answer synthesized from retrieved context.
    ```json
    {
      "llm_response": "هزینه ارسال سفارش بر اساس آدرس و شیوه انتخابی شما در مرحله نهایی سبد خرید مشخص می‌شود."
    }
    ```

---

## 📊 Evaluation Metrics

The evaluation module (`src/pipelines/evaluator/base.py`) computes standard information retrieval metrics against `test_digikala_faq.csv`:

- **MRR (Mean Reciprocal Rank)**:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
  Evaluates how early the first relevant category appears in the ranked results.
- **Precision@k**: Measures the proportion of relevant documents among the top-$k$ retrieved items.
- **Recall@k**: Measures the proportion of relevant documents retrieved out of all relevant documents.
- **Classification Report**: Precision, recall, and F1-score across all FAQ categories.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Here is the complete, professional summary with your ASCII architecture flowchart integrated perfectly into the design. I have enclosed the flowchart in a code block so that the spacing and alignment remain crisp and easy to read on GitHub or any text editor.

***

### SmartIngest: AI-Augmented Medallion Data Pipeline

**Project Overview**
SmartIngest is an end-to-end, production-style data engineering pipeline designed to ingest, process, and analyze heterogeneous data sources. The project strictly adheres to a Medallion architecture to ensure data quality and reliability, augmented by machine learning for anomaly detection and generative AI for automated health reporting.

**Architecture Flowchart**
```text
┌─────────────────────────────────────────────────────────────┐
│                       DATA SOURCES                          │
│  CSV (crop production) | JSON (hospitals) | REST API (weather)│
└──────────────┬─────────────────┬───────────────┬────────────┘
               │                 │               │
               ▼                 ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                     🥉 BRONZE LAYER                         │
│        Raw ingest → Parquet files + source metadata         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     🥈 SILVER LAYER                         │
│   Schema enforcement | Null handling | Dedup | Validation   │
│        Quality report: null %, drop rate, flags             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  🥇 GOLD LAYER  (DuckDB SQL)                │
│  GROUP BY | RANK() window fn | Rolling AVG | Cross-domain JOIN│
│       6 analytics tables saved as Parquet + CSV             │
└───────────────────┬─────────────────────┬───────────────────┘
                    │                     │
                    ▼                     ▼
     ┌──────────────────────┐  ┌─────────────────────┐
     │  🔍 Isolation Forest │  │  📄 Gemini AI Report │
     │  Anomaly detection   │  │  NL health summary   │
     └──────────┬───────────┘  └──────────┬──────────┘
                │                         │
                └────────────┬────────────┘
                             ▼
            ┌────────────────────────────────┐
            │   📊 HTML Dashboard (live)     │
            │   🔌 FastAPI REST endpoints    │
            │   📈 Streamlit Monitor         │
            └────────────────────────────────┘
```

**Detailed Architecture & Data Flow**
The system operates as a sequential, multi-layered pipeline that transforms raw data into business-ready analytics and serves it through live endpoints:

1.  **Data Sources (Ingestion):** Simultaneously extracts multi-format data across different domains, including structured files (CSV for crop data, JSON for hospital infrastructure) and live REST APIs (weather metrics).
2.  **Bronze Layer (Raw Storage):** Acts as the immutable landing zone. Data is ingested exactly as it arrives, tagged with source metadata, and efficiently stored in Apache Parquet format.
3.  **Silver Layer (Validation & Cleansing):** The quality control engine. This layer enforces strict schemas, handles missing values, deduplicates records, and calculates comprehensive data quality metrics (e.g., null percentages and drop rates) before saving the cleaned Parquet files.
4.  **Gold Layer (Aggregated Analytics):** The business logic layer. Utilizing DuckDB, it executes advanced SQL transformations directly on the Parquet files—such as window functions, rolling averages, and cross-domain joins—to produce finalized analytics tables.
5.  **Intelligence Layer (ML & AI):** An independent monitoring layer that evaluates pipeline health. A scikit-learn Isolation Forest model analyzes the Silver-layer quality metrics to detect data anomalies, while the Google Gemini API translates these metrics into natural-language health reports.
6.  **Serving Layer (Delivery & UI):** The finalized Gold data, ML anomaly scores, and AI reports are exposed via a FastAPI REST backend. This data powers a standalone HTML/Chart.js dashboard and a Streamlit monitoring interface for real-time visibility.

**Core Technology Stack**
* **Data Processing & Storage:** Python, Pandas, PyArrow, DuckDB, Apache Parquet
* **Machine Learning & AI:** scikit-learn (Isolation Forest), Google Gemini API
* **Backend & APIs:** FastAPI, Uvicorn
* **Monitoring & UI:** Streamlit, HTML/CSS/JS, Chart.js

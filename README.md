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

**How to Run the Project locally**

**1. Clone the repository**
```bash
git clone https://github.com/UDAYADITYA-2005/smartingest.git
cd smartingest
```

**2. Create and activate a virtual environment**
* **Windows:**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```
* **Mac / Linux:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**
Create a `.env` file in the root directory and add your Google Gemini API key to enable AI-generated reports. *(Note: The pipeline works without this using a built-in template fallback).*
```env
GEMINI_API_KEY=your_key_here
```

**5. Run the pipeline**
```bash
python pipeline.py
```

**6. Start the API server and view the dashboard**
```bash
uvicorn api:app --reload --port 8000
```
Once the server is running, open your web browser and navigate to `http://localhost:8000/dashboard` to view the live HTML dashboard.

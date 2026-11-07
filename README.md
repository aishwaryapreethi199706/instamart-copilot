# Instamart Customer Intelligence & Support Copilot

## Overview

This project builds a customer intelligence and support copilot for a simulated Swiggy Instamart business.

The project addresses four business problems:

1. **Data Foundation:** Audit, clean, validate and analyse raw customer, product and order data using Python, pandas and SQLite.
2. **Customer Segmentation and Lapse Prediction:** Segment customers using RFM analysis and K-Means clustering, then predict customer lapse using machine learning.
3. **Support Ticket Triage:** Redact sensitive information, classify support tickets, apply input guardrails, manage retries and token budgets, and route tickets to appropriate support teams.
4. **Policy RAG and Agent:** Retrieve information from policy documents and build a ReAct support agent capable of checking orders, calculating refunds, issuing eligible refunds and escalating cases.

The project uses the supplied offline `mock_llm.py` implementation. No paid LLM API is required.

## Reproduce

### Requirements

- Python 3.11 or later
- SQLite 3.25 or later
- Libraries listed in `requirements.txt`

### Setup and execution

Run the following commands from the repository root in Windows PowerShell:

1. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

2. Activate the environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Generate the supplied synthetic datasets:

   ```powershell
   python generate_data.py
   ```

5. Clean and audit the data:

   ```powershell
   python src/clean_data.py
   ```

6. Validate the cleaned data:

   ```powershell
   python src/validate_data.py
   ```

7. Load the SQLite database:

   ```powershell
   python src/load_database.py
   ```

8. Execute the SQL queries:

   ```powershell
   python src/run_queries.py
   ```

9. Run pandas analysis:

   ```powershell
   python src/pandas_analysis.py
   ```

10. Perform customer segmentation:

    ```powershell
    python src/customer_segments.py
    ```

11. Train and evaluate lapse prediction models:

    ```powershell
    python src/lapse_prediction.py
    ```

12. Run support ticket triage:

    ```powershell
    python src/ticket_triage.py
    ```

13. Run policy RAG evaluation:

    ```powershell
    python src/policy_rag.py
    ```

14. Run the ReAct support agent:

    ```powershell
    python src/agent.py
    ```

**Execution note:** `data/raw/agent_scenarios.csv` is a committed input used by the agent. Ensure it is present before running the final script. The supplied `generate_data.py` and `mock_llm.py` are retained unmodified.

On Windows systems experiencing numerical-library memory pressure during model training, the following environment settings can limit parallel numerical computation:

```powershell
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:LOKY_MAX_CPU_COUNT="1"
```

### Project structure

```text
instamart-copilot/
│
├── generate_data.py
├── mock_llm.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── customers.csv
│   │   ├── products.csv
│   │   ├── orders.csv
│   │   ├── tickets.csv
│   │   ├── policy_eval.csv
│   │   └── agent_scenarios.csv
│   │
│   ├── clean/
│   │   ├── customers.csv
│   │   ├── products.csv
│   │   └── orders.csv
│   │
│   ├── policies/
│   │   ├── delivery_policy.txt
│   │   ├── escalation_policy.txt
│   │   ├── payment_policy.txt
│   │   ├── privacy_policy.txt
│   │   └── refund_policy.txt
│   │
│   └── instamart.db
│
├── sql/
│   └── queries.sql
│
├── src/
│   ├── agent.py
│   ├── clean_data.py
│   ├── customer_segments.py
│   ├── lapse_prediction.py
│   ├── load_database.py
│   ├── pandas_analysis.py
│   ├── policy_rag.py
│   ├── run_queries.py
│   ├── ticket_triage.py
│   └── validate_data.py
│
├── outputs/
│   ├── customer_segments.csv
│   ├── kmeans_evaluation.csv
│   ├── rfm_table.csv
│   ├── lapse_model.joblib
│   ├── retention_list.csv
│   ├── triage_results.csv
│   ├── triage_budget_demo.csv
│   ├── triage_zero_shot.csv
│   ├── triage_few_shot.csv
│   ├── policy_rag_evaluation.csv
│   ├── policy_rag_zero_overlap.csv
│   ├── agent_runs.jsonl
│   └── refund_ledger.jsonl
│
└── state/
    ├── scenario_s1.json
    ├── scenario_s2.json
    ├── scenario_s3.json
    ├── scenario_s4.json
    ├── scenario_s5.json
    ├── scenario_s6.json
    ├── scenario_s7.json
    ├── test_max_iterations.json
    └── test_timeout.json
```

## Part A — Data Foundation

### 1. Raw data audit

The raw datasets were inspected for missing values, duplicates, inconsistent formats and invalid values.

| Dataset | Raw rows | Raw columns |
|---|---:|---:|
| Customers | 1,200 | 5 |
| Products | 40 | 4 |
| Orders | 13,084 | 9 |

#### Customer audit findings

- Missing or blank ages: 61
- Ages outside the 10–100 range: 6
- Distinct raw city spellings: 24
- Distinct non-blank gender spellings: 10
- Blank genders: 30

#### Order audit findings

- Exact duplicate rows: 119
- Pending status rows: 656
- Distinct order IDs: 12,309
- Non-numeric or missing order amounts after numeric conversion: 120
- Negative delivery times: 106
- Missing or blank delivery times after numeric conversion: 2,541

The missing-value counts and numeric-conversion counts are separate audit measurements and may overlap.

### 2. Cleaning rules

**Customers**

- Standardized city names by trimming whitespace and applying title case.
- Normalized gender values to `female` and `male`.
- Assigned `unknown` to blank or unmatched gender values.
- Converted age to numeric values.
- Treated ages outside 10–100 as missing.
- Filled missing ages using the median age of the customer's city.
- Converted signup dates to datetime values.

**Orders**

- Converted order timestamps to datetime values.
- Removed exact duplicate rows.
- Retained the latest timestamped record for each order ID.
- Standardized order status to lowercase and removed surrounding whitespace.
- Converted quantity, discount, order amount and delivery time to numeric values.
- Joined product category and product price.
- Filled missing order amounts using:

  `product price × quantity × (1 − discount percentage / 100)`

- Replaced negative delivery times with missing values.
- Filled missing delivery times for completed orders using the median delivery time of the corresponding product category.
- Preserved missing delivery times for cancelled orders.

### 3. Cleaned data results

**Customers**

| City | Customers |
|---|---:|
| Bengaluru | 197 |
| Chennai | 191 |
| Delhi | 210 |
| Hyderabad | 191 |
| Mumbai | 184 |
| Pune | 227 |
| **Total** | **1,200** |

Gender distribution:

- Female: 610
- Male: 560
- Unknown: 30

Final age range: 18–60. Missing ages: 0.

**Orders**

- Cleaned rows: 12,309
- Completed orders: 11,110
- Cancelled orders: 1,199
- Missing order amounts: 0
- Missing completed-order delivery times: 0
- Missing cancelled-order delivery times: 1,199
- Completed-order revenue: **INR 2,998,668.80**

### 4. Data validation

The validation script completed successfully:

`ALL CHECKPOINTS PASSED`

All nine validation checkpoints passed.

### 5. SQLite database and SQL analysis

The cleaned customer, product and order tables were loaded into `data/instamart.db`.

Seven SQL queries were implemented in `sql/queries.sql`.

| Query | SQL concept | Purpose | Result |
|---|---|---|---:|
| Q1 | WHERE and HAVING | Customers with qualifying completed orders and total spend above INR 8,000 | 30 rows |
| Q2 | Three-table JOIN | Revenue by city and product category | 30 rows |
| Q3 | Scalar subquery | Products priced above average product price | 11 rows |
| Q4 | LEFT JOIN | Customers without completed orders | 25 rows |
| Q5 | CTE | Customers spending above average customer spend | 550 rows |
| Q6 | Window functions | RANK, DENSE_RANK and ROW_NUMBER by city | 1,175 rows |
| Q7 | CTE and cumulative window function | Daily revenue and cumulative running total | 183 rows |

Q2 produced the following completed revenue by city:

| City | Revenue (INR) |
|---|---:|
| Bengaluru | 523,643.40 |
| Chennai | 463,629.15 |
| Delhi | 502,829.80 |
| Hyderabad | 528,589.90 |
| Mumbai | 479,058.50 |
| Pune | 500,918.05 |
| **Total** | **2,998,668.80** |

Q5 identified 550 customers whose completed-order spend was above the average customer spend of INR 2,552.06.

Q6 returned 1,175 customer rows and demonstrated the differences between the three ranking window functions. Eight rows had different RANK and ROW_NUMBER values.

Q7 covered 183 dates. The cumulative revenue reached INR 2,998,668.80 on the final date, 2025-09-30.

### 6. Pandas analysis

City-wise revenue and average order value:

| City | Revenue (INR) | Average order value (INR) | Orders |
|---|---:|---:|---:|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Order amount distribution:

| Statistic | Value (INR) |
|---|---:|
| Mean | 269.91 |
| Median | 135.00 |
| First quartile | 64.80 |
| Third quartile | 297.00 |
| Lower outlier fence | -283.50 |
| Upper outlier fence | 645.30 |
| Outlier count | 897 |

The order amount distribution is right-skewed, with a skewness of 4.61. The mean is higher than the median, indicating that a relatively small number of high-value orders influence the average.

#### Correlation analysis

| Feature pair | Correlation |
|---|---:|
| Quantity and order amount | +0.087 |
| Discount percentage and order amount | -0.095 |
| Delivery minutes and order amount | -0.003 |
| Discount percentage and quantity | Approximately -0.000 |
| Quantity and delivery minutes | -0.013 |
| Discount percentage and delivery minutes | -0.010 |

The strongest absolute correlation among these pairs is between discount percentage and order amount (-0.095). These are weak associations and should not be interpreted as evidence of causation.

## Part B — Customer Segmentation and Lapse Prediction

### 1. RFM analysis

RFM analysis was performed using:

- **Recency:** Time since the customer's most recent order.
- **Frequency:** Number of completed orders.
- **Monetary:** Total completed-order spending.

The RFM dataset contained 1,175 customers.

| RFM measurement | Value |
|---|---:|
| Median recency | 19 days |
| Mean frequency | 9.46 |
| Mean monetary value | INR 2,552.06 |

### 2. K-Means clustering

K-Means was evaluated for K values from 2 to 8.

| K | Inertia | Silhouette score |
|---:|---:|---:|
| 2 | 1820.76 | 0.465 |
| 3 | 1268.23 | 0.445 |
| 4 | 743.07 | 0.526 |
| 5 | 617.37 | 0.480 |
| 6 | 503.17 | 0.482 |
| 7 | 408.46 | 0.416 |
| 8 | 366.50 | 0.412 |

K=4 produced the highest silhouette score of 0.526 among the tested values.

### 3. Customer segment profiles

| Segment | Customers | Mean recency | Mean frequency | Mean monetary value (INR) |
|---|---:|---:|---:|---:|
| Regular Customers | 436 | 28.86 | 3.77 | 1,057.62 |
| Loyal Customers | 419 | 8.77 | 19.40 | 3,410.02 |
| At-Risk Customers | 182 | 109.04 | 2.20 | 685.99 |
| Premium Customers | 138 | 24.12 | 6.77 | 7,129.70 |
| **Total** | **1,175** | | | |

### 4. PCA

Principal Component Analysis was used to reduce the clustering feature space for analysis.

- First component explained variance: 63.4%.
- Second component explained variance: 21.8%.
- Combined explained variance: **85.2%**.

### 5. Lapse prediction cohort

The lapse prediction cohort contained 1,156 customers.

| Measurement | Result |
|---|---:|
| Cohort customers | 1,156 |
| Lapsed customers | 443 |
| Lapse rate | 38.32% |
| Feature dataset shape | 1,156 × 13 |
| Training rows | 924 |
| Test rows | 232 |

### 6. Model evaluation

A baseline model, Logistic Regression and Random Forest were evaluated.

**Baseline**

- Accuracy: 0.6164
- Lapsed-class F1: 0.0000

**Logistic Regression**

- Best parameter: `C = 100`
- Best cross-validation F1: 0.6506
- Test accuracy: 0.7284
- Precision: 0.6444
- Recall: 0.6517
- F1: 0.6480

Confusion matrix:

```text
[[111  32]
 [ 31  58]]
```

**Random Forest**

- Best parameters:
  - `max_depth = 8`
  - `min_samples_leaf = 10`
- Best cross-validation F1: 0.6811
- Test accuracy: 0.7371
- Precision: 0.6591
- Recall: 0.6517
- F1: 0.6554

Confusion matrix:

```text
[[113  30]
 [ 31  58]]
```

Random Forest achieved the highest test accuracy among the standard models and was selected as the saved model.

### 7. Oversampling

The training data was oversampled to balance the two lapse classes.

| Class | Training count after oversampling |
|---|---:|
| Not lapsed (0) | 570 |
| Lapsed (1) | 570 |

Oversampled Logistic Regression results:

- Accuracy: 0.7155
- Precision: 0.5891
- Recall: 0.8539
- F1: 0.6972

Confusion matrix:

```text
[[90 53]
 [13 76]]
```

Oversampling improved recall and F1 for the lapsed class, but reduced precision. This demonstrates the trade-off between identifying more potentially lapsed customers and generating additional false positives.

### 8. Feature importance

Top five Random Forest features:

| Feature | Importance |
|---|---:|
| Frequency | 0.279589 |
| Monetary | 0.271176 |
| Recency | 0.115149 |
| Average delivery minutes | 0.061286 |
| Cancellation rate | 0.052325 |

The most important features in the fitted Random Forest were frequency and monetary value.

### 9. Logistic Regression coefficients

Largest positive coefficients:

| Feature | Coefficient |
|---|---:|
| Monetary | 0.284573 |
| Average delivery minutes | 0.078514 |
| Recency | 0.039585 |

Largest negative coefficients:

| Feature | Coefficient |
|---|---:|
| Frequency | -1.698239 |
| Favourite category: Electronics | -1.136713 |
| Favourite category: Grocery | -0.448673 |

These coefficients describe the fitted model and should be interpreted in the context of its preprocessing and feature encoding.

### 10. Model saving and retention list

The selected Random Forest model was saved to:

`outputs/lapse_model.joblib`

The model was reloaded successfully, and predictions before and after reloading matched.

A retention list of the top 10 Premium customers by predicted lapse probability was saved to:

`outputs/retention_list.csv`

One of the ten customers in the displayed retention list was actually lapsed.

**Limitation:** The Premium segment flag is based on full-window RFM information, including September. This creates a post-hoc limitation when interpreting the retention list as a strictly prospective prediction.

## Part C — Support Ticket Triage

### 1. Privacy redaction

Sensitive information was redacted before classification.

| Sensitive pattern | Redactions |
|---|---:|
| Email addresses | 55 |
| Phone numbers | 91 |
| Order IDs | 132 |
| **Total** | **278** |

Residual sensitive patterns after redaction: **0**.

### 2. Input guardrails

| Input status | Count |
|---|---:|
| Empty tickets skipped | 4 |
| Long tickets flagged | 4 |
| Eligible tickets | 156 |

Long tickets were flagged for review rather than being treated as ordinary classification inputs.

### 3. Classification categories and prompts

The classifier uses four categories:

- Billing
- Delivery
- Technical
- Fraud

The system prompt requests a JSON-only response using the required classification schema.

Two prompt strategies were evaluated:

- **Zero-shot:** Classification without labelled examples.
- **Few-shot:** Classification with labelled examples, including fraud examples.

### 4. Output parsing and retries

Classification responses were parsed as JSON first.

When JSON parsing failed, a keyword fallback checked for category words in the response. A fallback was accepted only when exactly one valid category word was identified. Otherwise, the response was marked invalid.

Transient LLM errors were handled with a maximum of three retries and a short randomized backoff. Retry events were logged.

### 5. Zero-shot evaluation

- Eligible tickets: 156
- Successfully classified: 147
- Valid JSON outputs: 135
- Keyword fallbacks: 12
- Invalid outputs: 9
- Accuracy: **85.71%**
- Total tokens: 8,932
- Average tokens per classified ticket: 60.76

Confusion matrix (Billing, Delivery, Technical, Fraud):

```text
[[38  0  0  0]
 [ 0 36  0  0]
 [10  0 29  0]
 [11  0  0 23]]
```

### 6. Few-shot evaluation

- Eligible tickets: 156
- Successfully classified: 147
- Valid JSON outputs: 135
- Keyword fallbacks: 12
- Invalid outputs: 9
- Accuracy: **100%**
- Total tokens: 20,320
- Average tokens per classified ticket: 138.23

Confusion matrix:

```text
[[38  0  0  0]
 [ 0 36  0  0]
 [ 0  0 39  0]
 [ 0  0  0 34]]
```

Few-shot prompting achieved higher classification accuracy on the evaluated classified tickets, with increased token usage.

### 7. Token budget demonstration

A separate run tested a token budget of 6,000.

| Result | Count |
|---|---:|
| Classified | 99 |
| Budget exhausted | 51 |
| Invalid | 6 |
| Tokens used | 6,023 |

The budget is checked before each model call. A call that begins below the remaining budget can cause the final token total to exceed the configured limit.

### 8. Final classification and routing

Final ticket statuses:

| Status | Count |
|---|---:|
| Classified | 147 |
| Flagged invalid output | 9 |
| Skipped empty | 4 |
| Flagged long | 4 |
| **Total** | **164** |

Routing:

| Route | Tickets |
|---|---:|
| Support queue | 102 |
| Security team | 34 |
| Human review | 17 |
| Premium desk | 11 |
| **Total** | **164** |

Priority distribution:

| Priority | Tickets |
|---|---:|
| NORMAL | 119 |
| HIGH | 45 |
| **Total** | **164** |

The final results were saved to:

- `outputs/triage_results.csv`
- `outputs/triage_zero_shot.csv`
- `outputs/triage_few_shot.csv`
- `outputs/triage_budget_demo.csv`

## Part D — Policy RAG and ReAct Agent

### 1. Policy document chunking

Five policy documents were loaded and split into chunks.

| Policy document | Chunks |
|---|---:|
| Delivery | 3 |
| Escalation | 3 |
| Payment | 3 |
| Privacy | 2 |
| Refund | 4 |
| **Total** | **15** |

Chunk size: 40 words.

Chunk overlap: 10 words.

### 2. Retrieval test

The retrieval test ranked relevant policy chunks using similarity scores.

The top retrieved results for the refund query came from `refund_policy.txt`, followed by relevant refund-policy passages and a privacy-policy passage.

### 3. RAG evaluation

The evaluation contained 14 questions:

- 12 answerable questions.
- 2 unanswerable questions.

With 10-word overlap:

| Evaluation metric | Result |
|---|---:|
| Expected document found | 12/12 |
| Expected fact found | 12/12 |
| Correct fallback responses | 2/2 |

### 4. Zero-overlap comparison

A second evaluation was performed with no chunk overlap.

The zero-overlap configuration produced 13 chunks.

| Evaluation metric | 10-word overlap | Zero overlap |
|---|---:|---:|
| Expected document found | 12/12 | 12/12 |
| Expected fact found | 12/12 | 11/12 |
| Correct fallback responses | 2/2 | 2/2 |

The failed zero-overlap question concerned the time taken to return money when a payment failed but the amount was deducted.

The expected policy was `payment_policy.txt`, but the retrieved passages did not contain the complete expected fact.

This illustrates how chunk boundaries can separate related information and reduce retrieval completeness. The 10-word overlap configuration retrieved all 12 expected facts.

Evaluation outputs:

- `outputs/policy_rag_evaluation.csv`
- `outputs/policy_rag_zero_overlap.csv`

### 5. ReAct agent tools

The agent uses five tools:

1. `get_order` — retrieves order information.
2. `search_policy` — retrieves relevant policy information.
3. `calculate_refund` — calculates eligible refund amounts.
4. `issue_refund` — records an eligible refund.
5. `escalate_to_human` — escalates cases requiring human intervention.

The agent uses a ReAct reasoning and tool-execution loop with a maximum of six iterations. It also uses checkpoints and safety fallbacks.

### 6. Seven scenario evaluation

| Scenario | Actual result | LLM calls | Tokens |
|---|---|---:|---:|
| S1 | REFUND | 5 | 1,229 |
| S2 | REFUND | 5 | 1,229 |
| S3 | ESCALATE | 5 | 1,228 |
| S4 | DENIED | 4 | 875 |
| S5 | DENIED | 3 | 539 |
| S6 | NOT_FOUND | 2 | 206 |
| S7 | FALLBACK_MAX_ITERATIONS | 6 | 3,327 |

S1 and S2 issued eligible refunds of INR 128.25 and INR 135.00.

S3 escalated a refund above INR 1,000 to a human lead.

S4 denied a late-delivery refund because delivery was within 45 minutes.

S5 denied a refund because the order was cancelled.

S6 returned NOT_FOUND for an unknown order.

S7 reached the maximum iteration limit and returned `FALLBACK_MAX_ITERATIONS`.

**Scenario result:** S1–S6 matched their expected labels. S7's actual result was the more specific `FALLBACK_MAX_ITERATIONS`, while the expected scenario label was `FALLBACK`. This is recorded as a label mismatch rather than an exact match.

### 7. Checkpoint recovery

A simulated crash was introduced after the second tool call.

The agent resumed with this sequence:

```text
get_order
search_policy
calculate_refund
issue_refund
```

The resumed sequence matched the expected sequence.

`CHECKPOINT RESUME PASSED: True`

### 8. Safety fallback tests

The following fallback conditions were tested:

| Test | Result |
|---|---|
| Token limit | FALLBACK_TOKEN_LIMIT |
| Timeout | FALLBACK_TIMEOUT |
| Maximum iterations | FALLBACK_MAX_ITERATIONS |

All three safety fallback tests completed with the expected fallback types.

### 9. ReAct versus Plan-and-Execute

**ReAct** combines reasoning and action in an iterative loop. The agent decides what to do next based on the current observation, executes a tool and then uses the result to determine its next action. This is useful for customer support because order lookup, policy retrieval, refund calculation and escalation may depend on information discovered during execution.

**Plan-and-Execute** creates a plan before carrying out its steps. This can be useful for predictable workflows with known stages, but a fixed plan may need adjustment when unexpected results occur.

For this project, ReAct was selected because refund handling requires conditional decisions. For example, the agent must check order status and delivery time before calculating or issuing a refund. It must also escalate refunds above the permitted threshold and stop safely when limits are reached.

## Limitations and Next Steps

### Limitations

- The datasets are synthetic and do not represent live Instamart operational data.
- The offline mock LLM is used for controlled evaluation; results may differ from a production LLM.
- Few-shot accuracy was measured on the supplied evaluation data and should not be interpreted as a guarantee of real-world performance.
- The Premium retention flag uses full-window RFM information, including September, creating a post-hoc limitation for prospective interpretation.
- The refund ledger is a simulated record and does not initiate real financial transactions.
- The supplied agent scenario data does not provide a refund-request timestamp for independently verifying the seven-day refund request window.
- Scenario S7 returns `FALLBACK_MAX_ITERATIONS` instead of the expected generic label `FALLBACK`; the agent's safety fallback is documented explicitly.

### Next Steps

- Validate the models using additional real-world or independently generated data.
- Monitor lapse prediction performance over time and evaluate prospective retention outcomes.
- Improve ticket classification robustness while monitoring token consumption.
- Expand policy retrieval evaluation with more boundary-sensitive questions.
- Integrate the agent with authenticated production systems only after appropriate security, privacy and human-approval controls are established.
- Add monitoring, audit trails and human review for sensitive support and refund operations.

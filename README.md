# Instamart Customer Intelligence & Support Copilot

## Overview

This project builds an offline customer intelligence and support copilot for a simulated grocery-delivery business.

It addresses four business problems:

1. **Data Foundation:** Clean inconsistent customer and order data, load it into SQLite, validate data quality, and perform exploratory analysis.
2. **Customer Segmentation and Lapse Prediction:** Create RFM-based customer segments, identify lapsed customers, and compare machine-learning models.
3. **Support Ticket Triage:** Redact sensitive information, classify support tickets, apply input and output guardrails, and route tickets to the appropriate support team.
4. **Policy RAG and ReAct Agent:** Retrieve relevant policy passages, answer policy questions, calculate refunds, use tools, escalate high-value refunds, and recover from interrupted agent runs.

The project uses Python, pandas, SQLite, scikit-learn, TF-IDF, cosine similarity, and the supplied deterministic `mock_llm.py`.

All model interactions are offline. No paid API or external LLM service is required.

## Reproduce

Run the commands from the repository root.

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Generate the supplied synthetic dataset

```powershell
python generate_data.py
```

The supplied `generate_data.py` and `mock_llm.py` are used without modification.


### 3. Run Part A - Data Foundation

```powershell
python src\clean_data.py
python src\load_database.py
python src\run_queries.py
python src\validate_data.py
python src\pandas_analysis.py
```

### 4. Run Part B - Customer Intelligence

```powershell
python src\customer_segments.py
python src\lapse_prediction.py
```

### 5. Run Part C - Support Ticket Triage

```powershell
python src\ticket_triage.py
```

### 6. Run Part D - Policy RAG and ReAct Agent

```powershell
python src\policy_rag.py
python src\agent.py
```

The scripts generate their results in the `outputs/` directory. The agent also creates resumable checkpoints in `state/`.

## Part A - Data Foundation

### Data cleaning and database

- Cleaned customer and order datasets.
- Normalized city and categorical values.
- Loaded the cleaned data into SQLite.
- Created and executed seven SQL analysis queries.
- Ran nine validation checks.

**Database results:**

| Dataset | Rows |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Orders loaded into SQLite | 12,309 |

The validation script completed with `ALL CHECKPOINTS PASSED`.

Completed orders generated revenue of **INR 2,998,668.80**.

### Exploratory data analysis

- Completed orders: 11,110
- Cancelled orders: 1,199
- Mean completed order value: INR 269.91
- Median completed order value: INR 135.00
- Order-value distribution skewness: 4.61
- IQR-based high-value outliers: 897

The correlation between discount percentage and order amount was approximately -0.095. The correlation between quantity and order amount was approximately 0.087. These are weak associations and do not establish causation.

## Part B - Customer Segmentation and Lapse Prediction

### RFM segmentation

RFM features represent recency, frequency, and monetary value.

- Customers included in RFM analysis: 1,175
- K-means clusters: 4
- Silhouette score: approximately 0.526

**Choosing K.** I tested K from 2 to 8. Inertia falls quickly up to K=4 and more slowly after that (743.1 at K=4, 617.4 at K=5), so the elbow is at K=4. K=4 also has the highest silhouette score (0.526), and its four segments are easy for the business to act on. I chose K=4.

**Why inertia cannot pick K.** Inertia measures how close points are to their cluster centre. It always falls as K grows, so K=8 (366.5) is lower than K=2 (1820.8), because more clusters mean each point sits closer to a centre. A lower number alone therefore does not mean a better model, and we also need the elbow, the silhouette score and business meaning.

**Segment labels.** The cluster numbers (0, 1, 2, 3) are only labels and have no order. The Premium segment is the one with the highest mean monetary value, not a particular number.
- PCA explained variance: approximately 85.2%

**RFM summary (1,175 customers)**

| Measure | Minimum | Median | Mean | Maximum |
|---|---:|---:|---:|---:|
| Recency (days) | 0 | 19 | 33.56 | 180 |
| Frequency (orders) | 1 | 5 | 9.46 | 33 |
| Monetary (INR) | 12.00 | 2,286.50 | 2,552.06 | 14,766.55 |

The resulting customer segments were:

| Segment | Customers |
|---|---:|
| Regular | 436 |
| Premium | 138 |
| Loyal | 419 |
| At-Risk | 182 |

**Recommended action per segment**

| Segment | Customers | Recommended action |
|---|---:|---|
| Regular Customers | 436 | Send occasional offers to move them toward more frequent ordering. |
| Loyal Customers | 419 | Reward them with a loyalty programme and early access, so they stay. |
| At-Risk Customers | 182 | Send a win-back offer now, because they have not ordered for about 109 days on average. |
| Premium Customers | 138 | Give them a personal service line and priority support, because they have the highest spend (about INR 7,130). |

The Premium segment contains 138 customers.

### Lapse prediction

The lapse cohort contained 1,156 customers, of whom 443 were classified as lapsed, giving a lapse rate of 38.32%.

The dataset was split into 924 training rows and 232 test rows.

**Leakage controls.** No encoder, scaler or resampler saw test rows: all of them were fitted on the training rows only. No order dated 2025-09-01 or later was used to build the features. The segment labels from the K-means step were not used as features.

**Scaler check.** The scaler learned a mean frequency of 8.1147 from the 924 training rows, while the full-cohort mean is 8.0285. They differ because the scaler was fitted on the training rows only, and the test rows were left out of that calculation. A code check confirmed that the feature columns contain 0 missing values.

| Model | Test accuracy | Lapsed-class F1 |
|---|---:|---:|
| Baseline | 0.6164 | 0.0000 |
| Logistic Regression | 0.7284 | 0.6480 |
| Random Forest | 0.7371 | 0.6554 |
| Oversampled Logistic Regression | 0.7155 | 0.6972 |

**Model choice.** Random Forest was selected. It had the higher cross-validated F1 (0.6811 vs 0.6506) and a slightly higher test F1 on the lapsed class (0.6554 vs 0.6480), with 58 of the 89 truly lapsing customers caught by both models. Missing a lapsing customer means a lost customer with no retention offer, which costs far more than a wasted discount sent to a loyal customer who would have stayed. Because recall matters more than precision here, F1 on the lapsed class is a better guide than accuracy. The Random Forest also made fewer false alarms (30 vs 32 loyal customers wrongly flagged), so it was the better choice on both counts.

**Top 5 Random Forest features (importance) and business reading**

| Feature | Importance | Business reading |
|---|---:|---|
| Frequency | 0.2796 | How often a customer ordered is the strongest sign of whether they keep ordering. |
| Monetary | 0.2712 | Total spend before the cutoff is almost as important as frequency. |
| Recency | 0.1151 | The longer since the last order, the more likely the customer is drifting away. |
| Average delivery minutes | 0.0613 | Slower deliveries have some link to customers leaving. |
| Cancellation rate | 0.0523 | Customers who cancel more often are somewhat more likely to lapse. |

**Logistic Regression coefficients (sign) and business reading**

| Feature | Coefficient | Sign | Business reading |
|---|---:|---|---|
| Frequency | -1.6982 | Negative | More past orders means a lower chance of lapsing. |
| Favourite category: Electronics | -1.1367 | Negative | Customers who mostly buy electronics are less likely to lapse than the reference category. |
| Monetary | +0.2846 | Positive | Higher spend raises lapse risk only after frequency is accounted for, because spend and frequency move together. This is a model artefact, not a rule that big spenders leave. |

Oversampling increased lapsed-customer recall from approximately 0.6517 to 0.8539 and F1 from approximately 0.6480 to 0.6972, but reduced precision from approximately 0.6444 to 0.5891. This demonstrates the trade-off between identifying more lapsed customers and increasing false positives.

The selected model was saved and reloaded successfully, and its predictions matched after reloading.

**Retention list.** Among the test-set customers in the Premium segment, the 10 with the highest predicted lapse probability were saved to `outputs/retention_list.csv`. Of these 10 customers, 1 truly lapsed (C0525). The Premium flag comes from the whole-window RFM, which includes September, the month that defines the lapse label, so this retention list is illustrative and not a leakage-free evaluation.

**Evaluation limitation:** The Premium segment flag is based on the full-window RFM calculation, including the September period. It should therefore be treated as a descriptive customer segment rather than a strictly leakage-free predictive feature.

## Part C - Support Ticket Triage

The ticket triage pipeline includes:

- Email, phone number, and order ID redaction.
- Empty-input and long-input guardrails.
- Zero-shot and few-shot classification.
- Retry handling for simulated transient LLM failures.
- JSON validation and keyword-based fallback parsing.
- Invalid-output handling.
- Token-budget enforcement.
- Final routing by category, customer segment, and priority.

The four ticket categories are billing, delivery, technical, and fraud.

### Privacy and input validation

| Check | Result |
|---|---:|
| Email redactions | 55 |
| Phone redactions | 91 |
| Order ID redactions | 132 |
| Total redactions | 278 |
| Residual sensitive patterns | 0 |
| Empty tickets skipped | 4 |
| Long tickets flagged | 4 |
| Eligible tickets | 156 |

All 19 simulated transient errors were recovered through retries.

### Classification results

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Correct classifications | 126 | 147 |
| Accuracy | 85.71% | 100% |
| Average tokens per classified ticket | 60.76 | 138.23 |

Among the 156 eligible tickets:

- Zero-shot output: 135 valid JSON, 12 keyword-recovered, and 9 invalid.
- Few-shot output: 135 valid JSON, 12 keyword-recovered, and 9 invalid.

The zero-shot confusion matrix, using the order billing, delivery, technical, fraud, was:

```text
[[38,  0,  0,  0],
 [ 0, 36,  0,  0],
 [10,  0, 29,  0],
 [11,  0,  0, 23]]
```

### Budget and routing

The zero-shot budget demonstration used a token budget of 6,000. Processing stopped after 99 tickets when token usage reached 6,023, slightly exceeding the budget because the budget was checked before each classification.

Final triage output contained 164 rows:

| Route | Tickets |
|---|---:|
| support_queue | 102 |
| security_team | 34 |
| human_review | 17 |
| premium_desk | 11 |

The output includes ticket ID, customer ID, status, category, segment name, priority, route, and tokens used.

## Part D - Policy RAG and ReAct Agent

### Policy retrieval

The policy documents are split into 40-word chunks with an overlap of 10 words. TF-IDF vectorization and cosine similarity are used to retrieve the top three relevant passages.

| Policy document | Chunks |
|---|---:|
| delivery_policy.txt | 3 |
| escalation_policy.txt | 3 |
| payment_policy.txt | 3 |
| privacy_policy.txt | 2 |
| refund_policy.txt | 4 |
| **Total** | **15** |

Evaluation results:

- Expected document found: 12/12 answerable questions.
- Expected fact found: 12/12 answerable questions.
- Correct fallback responses: 2/2 unanswerable questions.

The zero-overlap comparison produced 11/12 expected facts. The question about the time taken to return money after a failed but deducted payment lost its expected fact in the retrieved passages. This illustrates how chunk overlap can help preserve information around chunk boundaries.

### ReAct agent

The agent uses five tools:

1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`
5. `escalate_to_human`

The agent uses a maximum of six iterations and supports token and time limits.

Seven supplied scenarios were executed:

| Scenario | Actual outcome |
|---|---|
| S1 | REFUND - INR 128.25 |
| S2 | REFUND - INR 135.00 |
| S3 | ESCALATE - INR 1,999.00 |
| S4 | DENIED |
| S5 | DENIED |
| S6 | NOT_FOUND |
| S7 | FALLBACK_MAX_ITERATIONS |

S7's expected CSV label is `FALLBACK`; the agent reports the more specific `FALLBACK_MAX_ITERATIONS`.

### Reliability and recovery

- LLM calls and token usage by scenario: S1: 5 calls/1,229 tokens; S2: 5/1,229; S3: 5/1,228; S4: 4/875; S5: 3/539; S6: 2/206; S7: 6/3,327.
- A simulated crash after the second tool execution was recovered from a saved checkpoint.
- The resumed run executed the remaining tools without repeating the first two. The complete tool sequence was `get_order`, `search_policy`, `calculate_refund`, and `issue_refund`.
- Token-limit test: `FALLBACK_TOKEN_LIMIT` after 2 calls and 242 tokens; only `get_order` executed.
- Timeout test: `FALLBACK_TIMEOUT` after 0 calls and 0 tokens; no tools executed.
- Maximum-iteration test: `FALLBACK_MAX_ITERATIONS` after 6 calls and 3,327 tokens.

The simulated refund actions are recorded in `outputs/refund_ledger.jsonl`, while scenario execution results are recorded in `outputs/agent_runs.jsonl`.

**Operational limitation:** All supplied simulated refund requests are treated as being within the policy's seven-day request window. The demonstration ledger records simulated refund actions; it does not connect to a real payment system.

## Limitations and Next Steps

- The project uses deterministic offline LLM simulation rather than a production language model.
- The synthetic dataset may not represent real-world customer behavior.
- Model performance should be evaluated on additional unseen data before production use.
- Customer segmentation and lapse prediction require periodic monitoring and retraining.
- The refund request date should be collected and validated before processing real refunds.
- Refund issuance and human escalation should be integrated with authenticated production systems before deployment.
- Further work could include a dashboard, monitoring, model drift detection, and a production-grade database.
- The Premium flag is a post-hoc whole-window label, because it uses RFM from the full window including September. The retention list is therefore illustrative and not a leakage-free evaluation.

## Project Structure

```text
instamart-copilot/
+-- generate_data.py
+-- mock_llm.py
+-- requirements.txt
+-- README.md
+-- .gitignore
+-- data/
|   +-- raw/        (generated by generate_data.py)
|   +-- clean/      (customers.csv, products.csv, orders.csv)
|   +-- policies/   (five policy text files)
|   +-- instamart.db
+-- sql/
|   +-- queries.sql
+-- src/
|   +-- audit_tables.py
|   +-- clean_data.py
|   +-- validate_data.py
|   +-- load_database.py
|   +-- run_queries.py
|   +-- pandas_analysis.py
|   +-- customer_segments.py
|   +-- lapse_prediction.py
|   +-- ticket_triage.py
|   +-- policy_rag.py
|   +-- agent.py
+-- outputs/        (segments, model, triage and agent outputs)
+-- state/          (agent checkpoints)
```

### ReAct vs Plan-Execute

The ReAct approach was selected because the support agent must make decisions based on information discovered during execution. It alternates between reasoning, tool calls, and observations. For example, it can look up an order, retrieve the relevant policy, calculate a refund, and then decide whether to issue the refund or escalate the case.

In the seven measured scenarios, the agent used 2–6 model calls and 206–3,327 tokens per run. Checkpoint recovery resumed after a simulated crash without repeating the first two tool executions. Token, timeout, and maximum-iteration fallbacks were also tested.

A Plan-Execute approach would create a sequence of actions in advance and then execute it. This can be more predictable for fixed workflows, while ReAct is more flexible when an order is missing, a refund is ineligible, or escalation is required. A separate Plan-Execute baseline was not implemented, so these results do not establish an empirical performance advantage. ReAct also requires iteration limits, token limits, timeout handling, and checkpoint recovery to control execution safely.

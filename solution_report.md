# AI Solution Design Report
## Healthcare Insurance — Medical Claim Fraud Detection System

**Domain:** Healthcare  
**AI Task Type:** Classification / Anomaly Detection  
**Prepared For:** Part 4 — AI Solution Design  

---

## Task 1: Business Domain

**Selected Domain: Healthcare**

Healthcare insurance fraud is one of the most financially damaging and operationally complex problems in the industry. In the United States alone, the FBI estimates fraudulent healthcare billing costs the industry over **$300 billion annually**. This makes it an ideal domain for an AI-driven solution — the problem is large-scale, data-rich, repeatable, and has measurable business outcomes.

---

## Task 2: Business Problem Definition

### What Problem Is Being Solved?

Medical insurance companies receive thousands of claims daily from hospitals, clinics, physicians, and patients. A significant fraction of these claims are **fraudulent** — involving overbilling, phantom procedures, upcoding (billing for a more expensive service than delivered), or identity theft. Identifying fraudulent claims manually is slow, inconsistent, and resource-intensive.

The AI solution proposes an automated **claim fraud scoring system** that assigns each submitted claim a fraud probability score in real time, routing suspicious claims to investigators and auto-approving low-risk ones.

### Who Are the Stakeholders?

| Stakeholder | Role |
|---|---|
| **Fraud Investigators** | Review flagged claims; provide feedback labels |
| **Claims Adjudicators** | Process approved claims; manage work queues |
| **Patients / Claimants** | Receive faster decisions on legitimate claims |
| **Healthcare Providers** | Submitters of claims; affected by holds |
| **Compliance Officers** | Ensure regulatory adherence (HIPAA, SOX) |
| **Data Science Team** | Build, monitor, and retrain models |
| **Insurance Executives** | Track fraud loss reduction and ROI |

### What Is the Current Manual Process?

Currently, claims are processed through a rules-based system supplemented by manual review:

1. Claim is submitted via provider portal or paper form
2. Basic eligibility checks run automatically (member active, procedure covered)
3. A subset of claims is selected for manual review using simple business rules (e.g. high dollar amount, certain procedure codes)
4. A fraud analyst reviews the selected claim, calls the provider, checks history
5. Decision made: approve, request more information, or deny
6. Investigation opened for suspected fraud

### Limitations of the Current Process

| Limitation | Impact |
|---|---|
| **Rule-based triggers miss novel fraud patterns** | Fraudsters adapt quickly to known rules |
| **Manual review is slow** (35–45 hours avg resolution) | Legitimate claimants face long waits |
| **High error rates** (avg 6.7% errors per KPI data) | Wrongful denials and approvals |
| **Low coverage** — only a fraction of claims reviewed | Most fraud passes undetected |
| **Inconsistent decisions** across different analysts | Unfair treatment of similar claims |
| **No learning** — rules don't improve over time | Static system in a dynamic fraud landscape |
| **High manual processing burden** (avg 450 hours/month per KPI data) | Operational cost |

---

## Task 3: AI Task Type

**Primary AI Task: Classification**  
**Secondary AI Task: Anomaly Detection**

### Why Classification?

Each claim can be labelled as one of two outcomes: **fraudulent** or **legitimate**. Historical labelled data (confirmed fraud investigations) exists, making this a supervised binary classification problem. The model learns patterns from past fraud cases and applies those patterns to score new claims.

### Why Anomaly Detection as Secondary?

Not all fraud is seen before. An autoencoder-based anomaly detector can flag claims that deviate significantly from normal patterns — catching **novel fraud schemes** that have no historical label.

### Why This Task Type Is Suitable

- Structured claim data (tabular features) maps directly to classification models
- Historical fraud labels are available from investigation outcomes
- Probability scores (rather than hard predictions) allow flexible threshold tuning
- The combination of supervised + unsupervised detection creates a robust two-layer defence

---

## Task 4: Data Requirement Plan

### Type of Data Needed

**Structured data** (primary) — tabular claim records with categorical and numerical features.

### Input Features

| Feature Category | Features |
|---|---|
| **Claim details** | Claim amount, procedure codes (CPT), diagnosis codes (ICD-10), claim date, service date |
| **Provider features** | Provider ID, specialty, geographic region, years in network, historical claim volume |
| **Patient features** | Age, gender, insurance plan type, tenure, prior claims count |
| **Billing patterns** | Claims per patient per month, avg claim amount vs. specialty benchmark |
| **Anomaly signals** | Duplicate procedure count, weekend/holiday billing flag, unusual time-to-submit |
| **Network features** | Provider–patient pair frequency, referral chain depth |
| **Historical ratios** | Provider's historical fraud rate, patient's prior investigation flag |

### Target Variable

- **`fraud_label`** — Binary: `1 = fraudulent`, `0 = legitimate`
- Derived from confirmed investigation outcomes stored in the claims management system

### Data Collection Method

| Source | Method |
|---|---|
| Claims Management System | Direct database export (nightly batch + real-time stream) |
| Investigation Outcomes | Manual label extraction from case management system |
| Provider Registry | API integration with provider credentialing system |
| EHR Integration | HL7 FHIR API for clinical context (optional enrichment) |

### Data Quality Risks

| Risk | Mitigation |
|---|---|
| **Class imbalance** (~1–3% fraud rate) | SMOTE oversampling + cost-sensitive loss |
| **Label noise** — investigations sometimes wrong | Confidence-weighted labels; human review loop |
| **Missing values** — incomplete claims | Median/mode imputation; missingness as feature |
| **Temporal leakage** — future data in training | Strict time-based train/test split |
| **Concept drift** — fraud tactics change | Monthly retraining; drift detection alerts |
| **Privacy risk** — patient data sensitivity | Anonymisation, differential privacy, HIPAA compliance |

---

## Task 5: Model Recommendation

### Recommended Architecture: Ensemble of Feed-Forward Neural Network + Gradient Boosting

#### Primary Model — Feed-Forward Neural Network (FFNN)

```
Input Layer     : 32 engineered features
Hidden Layer 1  : Dense(256) + ReLU + BatchNorm + Dropout(0.3)
Hidden Layer 2  : Dense(128) + ReLU + BatchNorm + Dropout(0.3)
Hidden Layer 3  : Dense(64)  + ReLU
Output Layer    : Dense(1)   + Sigmoid  →  fraud probability [0,1]

Loss            : Binary Cross-Entropy (weighted for class imbalance)
Optimizer       : Adam (lr=0.001)
Regularisation  : L2 weight decay + Dropout
```

#### Secondary Model — XGBoost (Gradient Boosting)

- Handles class imbalance natively with `scale_pos_weight`
- Provides interpretable feature importances
- More robust to missing values and feature scaling
- Complements FFNN by capturing non-linear tree-based interactions

#### Ensemble Strategy

```
Final Score = 0.6 × FFNN_probability + 0.4 × XGBoost_probability
Decision threshold : 0.45  (tuned to maximise recall at ≥85%)
```

#### Why This Architecture Is Appropriate

| Reason | Justification |
|---|---|
| **FFNN learns complex interactions** | Fraud signals are rarely linear — deep layers capture compound patterns |
| **XGBoost adds robustness** | Tree models complement neural networks on tabular data |
| **Ensemble reduces variance** | Averaging two independent models improves stability |
| **Sigmoid output** | Natural probability interpretation for threshold tuning |
| **Batch Normalisation** | Stabilises training on features with different scales |
| **SMOTE + weighted loss** | Addresses the severe class imbalance in fraud datasets |

#### Explainability — SHAP Values

SHAP (SHapley Additive exPlanations) values are computed for every flagged claim, showing investigators exactly which features drove the fraud score. This is essential for:
- Regulatory compliance (right to explanation)
- Investigator trust and adoption
- Identifying model weaknesses

---

## Task 6: Evaluation Plan

### Technical Metrics

| Metric | Target | Rationale |
|---|---|---|
| **Precision** | ≥ 90% | Minimise false accusations of legitimate providers |
| **Recall** | ≥ 85% | Catch most fraudulent claims |
| **F1-Score** | ≥ 0.87 | Balanced trade-off |
| **AUC-ROC** | ≥ 0.95 | Overall discriminative power |
| **False Positive Rate** | ≤ 5% | Limit wrongful claim holds |
| **Average Precision (PR-AUC)** | ≥ 0.80 | Precision-recall trade-off under imbalance |

### Business Metrics

| Metric | Baseline (KPI Data) | Target |
|---|---|---|
| **Manual processing hours/month** | ~450 hours | ↓ 60% to ~180 hours |
| **Average resolution time** | 35–45 hours | ↓ to < 15 hours (legitimate claims) |
| **Error rate** | 6.7% | ↓ to < 2% |
| **Customer satisfaction (CSAT)** | 6.9 / 10 | ↑ to 8.0 / 10 |
| **Fraud loss reduction** | Baseline | ↓ 30–40% in first year |
| **Investigator case efficiency** | — | ↑ 2× cases reviewed per analyst |

### Possible Failure Cases

| Failure | Likelihood | Impact |
|---|---|---|
| **False positive** — legitimate claim flagged | Medium | Patient/provider frustration; payment delay |
| **False negative** — fraud missed | Low-Medium | Financial loss; missed fraud ring |
| **Model drift** — fraud tactics change | Medium-High over time | Accuracy degrades; old model becomes ineffective |
| **Adversarial gaming** — fraudsters learn model signals | Low initially | Precision drops; retraining required |
| **Data pipeline failure** — stale features | Low | Incorrect real-time scores |

### Human Review and Validation Process

1. **All high-risk claims (score ≥ 0.45)** are reviewed by a trained fraud investigator
2. **Investigator outcome** (confirmed fraud / false alarm) fed back as new training label
3. **Monthly model review meeting** — data science team reviews precision/recall trends
4. **Quarterly external audit** — independent review of model decisions for bias
5. **Shadow mode deployment** — new model versions run in parallel for 30 days before cutover

---

## Task 7: Responsible AI Considerations

### Bias in Data

Historical fraud labels reflect which claims investigators chose to investigate — often influenced by existing biases (e.g. smaller providers investigated more often, certain demographic groups flagged more). The model trained on these labels can **inherit and amplify** investigator bias.

**Mitigation:** Fairness-aware training; bias audits across provider geography, specialty, and patient demographics; disparate impact testing before deployment.

### Incorrect Predictions

A false fraud flag can delay payment to a legitimate patient in financial distress or damage a physician's reputation. A missed fraud case results in financial loss and enables continued criminal behaviour.

**Mitigation:** Conservative thresholds tuned toward reducing false positives; mandatory human review for all flagged claims; appeal mechanism for wrongly flagged providers.

### Privacy Concerns

Claim data contains highly sensitive personal health information (PHI) protected under HIPAA in the US and GDPR in Europe. Breach or misuse could cause serious patient harm.

**Mitigation:** Data anonymisation and pseudonymisation; differential privacy during model training; access controls and audit logging; data retention policies; vendor contracts with BAAs (Business Associate Agreements).

### Over-Reliance on AI

Investigators may over-trust model scores and stop applying independent judgement, leading to systematic errors when the model is wrong.

**Mitigation:** SHAP explanations to encourage reasoning rather than score-following; regular calibration exercises; clear communication that the model is a decision-support tool, not a decision-maker.

### Impact on Users

- **Patients:** May experience payment delays if wrongly flagged; no visibility into why
- **Providers:** May be wrongly suspected, damaging professional reputation
- **Investigators:** May face deskilling if AI handles most decisions

**Mitigation:** Transparent appeals process; provider notification and explanation letters citing specific concerns (not model score); training programs for investigators to maintain domain expertise.

### Need for Human Oversight

All claims flagged as high-risk **must** be reviewed by a human investigator before any adverse action (denial, investigation, provider suspension). The AI system is deployed as a **decision-support tool**, not an autonomous decision-maker. This is a regulatory requirement in most jurisdictions for consequential financial decisions.

---

## Task 8: Final Solution Summary

---

### One-Page Solution Summary

**Problem**  
Healthcare insurance companies lose an estimated 3–10% of total claim spend to fraud annually. The current rules-based manual review process covers only a fraction of submitted claims, resolves cases in 35–45 hours on average, and has a 6.7% error rate — leaving significant fraud undetected and legitimate claimants waiting.

**Proposed AI Solution**  
An ensemble AI fraud detection system combining a Feed-Forward Neural Network and XGBoost classifier. Each submitted claim is scored in real time (fraud probability 0–1). Claims below 0.30 are auto-approved; claims above 0.45 are routed to investigators with SHAP-based explanations. Claims between 0.30–0.45 enter a soft-review queue with document requests.

**Required Data**

| Category | Source |
|---|---|
| Claim records (amount, codes, dates) | Claims Management System |
| Provider history & registry | Provider Credentialing API |
| Patient history & demographics | EHR / Member Database |
| Investigation outcomes (labels) | Case Management System |

**Model Recommendation**  
Ensemble: FFNN (256→128→64→1, Sigmoid) + XGBoost  
Combined via weighted average (60/40). SMOTE applied for class imbalance. SHAP for explainability.

**Expected Business Impact**

| KPI | Improvement |
|---|---|
| Fraud loss reduction | 30–40% year 1 |
| Manual processing hours | ↓ 60% |
| Resolution time (legitimate) | ↓ from 35h to <15h |
| Error rate | ↓ from 6.7% to <2% |
| Investigator throughput | ↑ 2× claims per analyst |
| Customer satisfaction | ↑ from 6.9 to 8.0/10 |

**Risks and Mitigation Plan**

| Risk | Mitigation |
|---|---|
| Investigator bias in labels | Fairness audits; demographic parity testing |
| False fraud flags on legitimate claims | Mandatory human review; appeals process |
| HIPAA/privacy violations | Anonymisation; differential privacy; access controls |
| Model drift over time | Monthly retraining; automated drift monitoring |
| Over-reliance on AI | SHAP explanations; AI as decision-support only |

---

*Report prepared for Part 4 — AI Solution Design.*  
*Dataset reference: `ai_usecase_reference_catalog.csv`, `business_kpi_sample.csv`*

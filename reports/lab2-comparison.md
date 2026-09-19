# Lab 2 — Run comparison

Experiment `itcs355-lab2` · 13 trials · total spend 0.0017 THB

`thb_per_point` is cost per percentage point of val_roc_auc above the worst trial. Cheap improvements rank low; expensive improvements rank high, however good the headline number is.

| run_id   |   val_roc_auc |   cost_thb |   n_estimators |   max_depth |   min_samples_leaf |   thb_per_point |
|:---------|--------------:|-----------:|---------------:|------------:|-------------------:|----------------:|
| c491f4f8 |        0.8432 |     0.0001 |            100 |           4 |                  3 |          0      |
| 8e063514 |        0.8426 |     0.0003 |            100 |           4 |                  5 |          0.0001 |
| 262ffe60 |        0.8424 |     0.0001 |            100 |           4 |                  1 |          0      |
| f0f35f01 |        0.8415 |     0.0001 |             50 |           4 |                  5 |          0      |
| d7055f8f |        0.8408 |     0.0001 |             50 |           4 |                  3 |          0      |
| 9a9927ca |        0.8405 |     0.0002 |             50 |           4 |                  1 |          0.0001 |
| 2649ad76 |        0.8394 |     0.0002 |             50 |           8 |                  3 |          0.0001 |
| 639bcfd4 |        0.8394 |     0.0001 |             50 |           8 |                  3 |          0      |
| a9da0925 |        0.8381 |     0.0001 |             50 |           8 |                  5 |          0      |
| a922cc3a |        0.8325 |     0.0001 |             50 |          14 |                  3 |          0.0001 |
| 4f24008d |        0.8307 |     0.0001 |             50 |          14 |                  5 |          0.0001 |
| 36059ad1 |        0.827  |     0.0001 |             50 |           8 |                  1 |          0.0001 |
| e2c41348 |        0.8149 |     0.0001 |             50 |          14 |                  1 |       1000      |

## Which model did you register, and why?
- I selected the 50-tree model `(f0f35f01: n_estimators=50, max_depth=4, min_samples_leaf=5)`achieving 0.8415 val ROC-AUC, rather than the highest-scoring 100-tree trial `(c491f4f8: n_estimators=100, max_depth=4, min_samples_leaf=3)`. The `0.0017` margin between two models is well within the seed variance measured across five random seeds (mean 0.855, std ±0.013), this means the gap between two models is not significant, it have no real meaning, it just random noise. Choosing 50 trees cuts model size and inference latency in half.
- Acorss five random seeds, the highest score is 0.8565, lowest is 0.8439, average is 0.8509. The difference between highest and lowest is 0.0126. so 0.0017 is very small compared to that.
- Training took ~0.1 seconds, costing ~0.0001 THB under Azure Spot (30% discount from normal price) pricing on `Standard_DS3_v2`. My assumptions for Azure cost are: an automated monthly retraining pipeline on Azure ML with job overhead (2 minutes at Spot rate 3.0 THB/hr verified on Azure calculator), retraining costs under 0.10 THB monthly.
- This choice could be wrong if dataset grows and develops complex non-linear failure modes requiring higher tree depth (such as depth 8 or 14) that were underrepresented in the current training window.

---

## Regional Price Verification (Azure East Asia vs Starting Values)

Verified against the [Azure Pricing Calculator](https://azure.microsoft.com/en-us/pricing/calculator/) and Azure Retail Prices API on September 17, 2026 (conversion rate: 1 USD ≈ 34.5 THB):ß

| Instance Type | Specs | On-Demand (USD) | On-Demand (THB/hr) | Spot (USD) | Spot (THB/hr) | `costs.py` Table |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard_DS3_v2** | 4 vCPU, 14 GB RAM | $0.428 / hr | **~14.77 THB** | $0.0871 / hr | **~3.01 THB** | 8.1 THB (US East baseline) |
| **Standard_F4s_v2** | 4 vCPU, 8 GB RAM | $0.216 / hr | **~7.45 THB** | $0.0399 / hr | **~1.38 THB** | 6.9 THB |
| **Standard_NC4as_T4_v3** | 4 vCPU, 28 GB, 1x T4 GPU | $0.736 / hr* | **~25.39 THB** | $0.2087 / hr | **~7.20 THB** | 24.5 THB |

*\*Note: Azure provides NC-series GPU VMs in nearby Southeast Asia (Singapore), Japan, and Korea datacenters; they are not provisioned directly in the East Asia (Hong Kong) datacenter.*

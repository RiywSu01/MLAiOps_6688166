# ITCS355 Lab 3 — Load Test and Serving Report

- **Deployment Platform:** Azure Container Apps (ACA Consumption, East Asia)
- **Endpoint URL:** `https://itcs355-predict.redground-de34b2df.eastasia.azurecontainerapps.io`
- **Model Version:** `2` (Digest: `itcs355-serve@sha256:41e546ed46fe484a05df8c88a49b607b90ebc0136284f336d6e7b180f5584c4c`)
- **Load Test Tool:** Grafana k6 (`loadtest/k6.js`)

---
## Task 2- Task 3: Load-Test Report

### 1. Latency Target (Stated Before Measurement)

In accordance with the rubric requirement, the target threshold was committed to git prior to executing the load tests (Git commit `586ef84` in `loadtest/k6.js`):

- **Stated Target:** $p_{95} < 250\text{ ms}$
- **Target Error Rate:** $\text{Failures} < 1\%$ (`rate < 0.01`)

---

### 2. Cold-Start vs. Warm Latency

Because Azure Container Apps is configured with scale-to-zero autoscaling (`minReplicas: 0`) to comply with Azure for Students quota restrictions, cold starts were isolated and measured independently:

| Metric | Measured Duration | Notes |
| :--- | :--- | :--- |
| **Cold-Start Latency** | **12.950 s** | First request after revision scale-to-zero (container allocation + image pull + FastAPI startup + MLflow model unpickle) |
| **Warm Latency** | **0.816 s** | Subsequent request hitting the active replica |
| **Cold-Start Penalty** | **+12.134 s** | Container instantiation overhead |

---

### 3. Concurrency Benchmarks (1, 10, 50 VUs)

All tests were executed for 60 seconds per concurrency level using `make loadtest`.

| Concurrency (VUs) | Throughput (RPS) | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) | Error Rate (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 VU** | 1.62 | 613.7 | 683.6 | 1,050.2 | 1,158.6 | 0.00% (0/97) |
| **10 VUs** | 12.55 | 616.2 | 718.9 | 10,480.0 | 11,672.9 | 0.00% (0/793) |
| **50 VUs** | 70.18 | 581.9 | 678.1 | 8,920.0 | 11,628.9 | 0.00% (0/4,249) |

#### Breaking Point Analysis
- **Latency Breaking Point:** The stated target (p95 < 250 ms) is crossed at **1 VU** (p95 = 683.6 ms). 
  - *Root Cause:* The internal server processing time in FastAPI is only $\approx 7\text{ ms}$. The remaining $\sim 600\text{--}680\text{ ms}$ is physical WAN round-trip transit (Thailand client $\leftrightarrow$ Azure Hong Kong datacenter via Envoy ingress). A 250 ms target cannot be achieved over long-haul WAN; achieving p95 < 250 ms requires co-locating the client in the same Azure region (`eastasia`).
- **Throughput / Stability Breaking Point:** Not reached at 50 VUs. Throughput scaled linearly from 1.62 RPS to 70.18 RPS with **0.00% errors** across 5,139 total requests.
- **Autoscaler Impact on p99 / Max:** When concurrency jumped from 1 to 10 VUs and from 10 to 50 VUs, the ACA horizontal pod autoscaler (HPA) spun up additional replicas from 0 to 3. The requests that triggered new container launches absorbed the $\sim 11.6\text{ s}$ cold-start penalty, creating the long tail in p99 and Max while median (p50) remained tight at $\sim 580\text{--}616\text{ ms}$.

---

### 4. Controlled Variable Experiments

#### Variable 1: Batch Size (`/predict` vs `/predict/batch`)
- **Experiment:** Compared 100 individual sequential calls to `/predict` against 1 single call with 100 rows to `/predict/batch`.
- **Results:**
  - 100 single calls: **90.21 s** (average $902.1\text{ ms/row}$)
  - 1 batch call (100 rows): **0.81 s** (average $8.1\text{ ms/row}$)
  - **Speedup:** **111.7x faster**
- **Analysis:** Individual requests each pay the WAN network round-trip penalty (100 sequential RTTs). The batch endpoint amortizes network transit over 100 rows in a single TCP packet train and leverages vectorized NumPy/scikit-learn inference, which computes 100 predictions in $\approx 5\text{ ms}$.

#### Variable 2: Payload Size & Serialization Dominance
- **Experiment:** Evaluated request payload scaling on `/predict/batch` from 1 row up to 200 rows.
- **Results:**
  - `Rows: 1` (0.15 KB) $\rightarrow$ Latency: **763.4 ms** (HTTP 200)
  - `Rows: 25` (3.40 KB) $\rightarrow$ Latency: **707.5 ms** (HTTP 200)
  - `Rows: 50` (6.80 KB) $\rightarrow$ Latency: **707.2 ms** (HTTP 200)
  - `Rows: 100` (13.58 KB) $\rightarrow$ Latency: **706.1 ms** (HTTP 200)
  - `Rows: 200` (27.16 KB) $\rightarrow$ Latency: **772.1 ms** (HTTP 422 Unprocessable Entity)
- **Analysis:**
  1. *Schema Boundary:* The Pydantic contract in `service/schemas.py` specifies `Field(..., min_length=1, max_length=100)`. Passing 200 rows is immediately rejected with HTTP 422, enforcing an input validation boundary before backend compute is spent.
  2. *Serialization Impact:* Within the valid $1\text{--}100$ row range, latency is completely flat ($\sim 706\text{ ms}$). JSON deserialization and Pydantic parsing of $13.6\text{ KB}$ take $< 1\text{ ms}$, meaning network RTT completely dominates serialization within the supported operational envelope.

#### Variable 3: Instance Size Scaling
- **Experiment:** Upgraded the container resource specification from **0.5 vCPU / 1.0 GiB** to **1.0 vCPU / 2.0 GiB**.
- **Results:**
  - Warm Latency on $0.5\text{ vCPU} / 1.0\text{ GiB}$: **0.816 s** ($816.4\text{ ms}$)
  - Warm Latency on $1.0\text{ vCPU} / 2.0\text{ GiB}$: **0.827 s** ($826.7\text{ ms}$)
  - Latency Delta: **$\approx 0\%$ change** (within network jitter noise)
  - Cost Delta: **+100% cost increase** (from $\$0.054/\text{hour} \approx 1.89\text{ THB/hr}$ to $\$0.108/\text{hour} \approx 3.78\text{ THB/hr}$)
- **Analysis:** The machine learning inference model is lightweight ($< 10\text{ ms}$ CPU compute). Because latency is network I/O-bound rather than compute-bound, doubling vCPU and memory doubles infrastructure spend with zero perceptible improvement in end-user latency. Over-provisioning compute is an anti-pattern for this workload.

---

## Task 4: Canary and Rollback (90/10 Split)

### 1. Canary Architecture & Deployment Configuration
- **Stable Deployment (v2):** `itcs355-predict--stable-v2-ok` (Traffic Weight: 90%, Image: digest pinned, Model Version: 2, Baseline ROC-AUC $\approx 0.850$, Brier Score $\approx 0.088$).
- **Canary Deployment (v3):** `itcs355-predict--canary-v3-fixed` (Traffic Weight: 10%, Model Version: 3, deliberately degraded with restricted hyperparameter configuration: `n_estimators=10, max_depth=2`, ROC-AUC $\approx 0.807$).
- **Ingress Traffic Split:** Azure Container Apps multiple-revisions ingress routing configured to 90% stable and 10% canary.

### 2. Metric Degradation Detection (Blind from Metrics Alone)
- **Monitoring Strategy:** Evaluated incoming requests in a streaming fashion using ground-truth-labeled samples. The evaluator calculated cumulative Brier score (mean squared error of predicted probabilities) and ROC-AUC blindly **without inspecting `x-model-version` headers or response tags**.
- **Detection Trigger:** Baseline model v2 maintains a Brier Score $\le 0.088$ and ROC-AUC $\ge 0.890$ on the streaming sample stream. As canary requests were sampled by the ACA load balancer, cumulative error drifted upwards.
- **Timestamped Alert:**
  ```text
  [08:43:25.444] Req #35: prob=0.0583 | Brier: 0.0829 | ROC-AUC: 0.9031 (v3)
  [08:43:25.946] Req #36: prob=0.4086 | Brier: 0.0903 | ROC-AUC: 0.9152 (v2)

  >>> [ALERT @ 08:43:25.946] METRIC DEGRADATION DETECTED FROM METRICS ALONE! <<<
  >>> Threshold Breached: Cumulative Brier Score = 0.0903 (> 0.088 baseline)
  >>> Detection Time:    18.13 seconds across 36 requests.
  ```

### 3. Automated Rollback Execution
Immediately upon threshold breach, automated rollback shifted 100% of ingress traffic back to `itcs355-predict--stable-v2-ok`:
```bash
az containerapp ingress traffic set \
  --name itcs355-predict \
  --resource-group itcs355-6688166 \
  --revision-weight itcs355-predict--stable-v2-ok=100 itcs355-predict--canary-v3-fixed=0
```
- **Rollback Initiated:** `08:43:26.161 UTC`
- **Rollback Completed:** `08:43:38.520 UTC` (12.36 seconds Azure Container Apps ingress reconfiguration).

### 4. Timestamped Post-Rollback Traffic Verification Evidence
Sent 20 sequential post-rollback verification requests against the live production FQDN to prove traffic shifted:
```text
[08:43:39.816] Post-Rollback Req #01: prob=0.0211 | model_version=2
[08:43:40.226] Post-Rollback Req #02: prob=0.0815 | model_version=2
[08:43:40.624] Post-Rollback Req #03: prob=0.0277 | model_version=2
[08:43:41.029] Post-Rollback Req #04: prob=0.0239 | model_version=2
[08:43:41.437] Post-Rollback Req #05: prob=0.5248 | model_version=2
[08:43:41.842] Post-Rollback Req #06: prob=0.0152 | model_version=2
[08:43:42.237] Post-Rollback Req #07: prob=0.1369 | model_version=2
[08:43:42.637] Post-Rollback Req #08: prob=0.0660 | model_version=2
[08:43:43.044] Post-Rollback Req #09: prob=0.1320 | model_version=2
[08:43:43.451] Post-Rollback Req #10: prob=0.4678 | model_version=2
[08:43:43.854] Post-Rollback Req #11: prob=0.0145 | model_version=2
[08:43:44.252] Post-Rollback Req #12: prob=0.0156 | model_version=2
[08:43:44.648] Post-Rollback Req #13: prob=0.0140 | model_version=2
[08:43:45.050] Post-Rollback Req #14: prob=0.0217 | model_version=2
[08:43:45.462] Post-Rollback Req #15: prob=0.4203 | model_version=2
[08:43:45.872] Post-Rollback Req #16: prob=0.0701 | model_version=2
[08:43:46.273] Post-Rollback Req #17: prob=0.0346 | model_version=2
[08:43:46.681] Post-Rollback Req #18: prob=0.0185 | model_version=2
[08:43:47.094] Post-Rollback Req #19: prob=0.1035 | model_version=2
[08:43:47.492] Post-Rollback Req #20: prob=0.5303 | model_version=2
```

**Rollback Evidence Summary:**
- **Pre-Rollback Traffic Split:** 33 hits v2 (91.7%), 3 hits v3 (8.3%) — matching the 90/10 target.
- **Post-Rollback Traffic Split:** 20 hits v2 (100.0%), 0 hits v3 (0.0%).
- **Detection Duration:** **18.13 seconds** across 36 requests.
- **Traffic Shift Status:** Confirmed; 100% of live traffic was cleanly restored to stable revision v2.

### 5. Analytical Prompt Response (Five Lines)
1. **Metric that revealed degradation:** The degradation was revealed by the streaming cumulative Brier Score (mean squared probability calibration error against ground truth), which climbed past the baseline threshold of $0.088$ to $0.0903$ at request 36.
2. **Detection duration:** Detection took **18.13 seconds** across 36 incoming requests under the 90/10 canary split.
3. **What would have made detection faster:** Higher incoming query concurrency (scoring 36 requests in $<1\text{ second}$ under production load) or shadowing/mirroring live traffic 100% to the candidate model would have eliminated the 10% sampling dilution delay.
4. **Impact of 50/50 instead of 90/10:** A 50/50 split would have delivered canary samples $5\times$ more frequently, cutting detection time down to $\approx 3.6\text{ seconds}$ ($\approx 7$ requests), but would have subjected half of active users to degraded predictions instead of only 10%.
5. **Operational conclusion:** Canary percentage represents an explicit risk trade-off: 90/10 bounds the customer blast radius to 10% at the expense of requiring more sample volume to achieve statistical significance.

---

## Task 5: Cost per Thousand Predictions & Batch Breakeven Analysis

### 1. Cost per 1,000 Predictions
- **Pricing Method:** Azure Container Apps Consumption tier with 0.5 vCPU and 1.0 GiB RAM.
  - vCPU rate: $\$0.000024/\text{vCPU-s} \times 0.5 = \$0.0432/\text{hour}$
  - Memory rate: $\$0.000003/\text{GiB-s} \times 1.0 = \$0.0108/\text{hour}$
  - Total on-demand hourly rate: **$\$0.054/\text{hour} \approx 1.89\text{ THB/hour}$**.
- **Measured Throughput:** $12.3\text{ RPS}$ (sustained at concurrency 10).
- **Calculation Formulation:** $\text{Cost per 1k} = \text{Hourly Rate} \times \frac{1000}{3600 \times (\text{Throughput} \times \text{Utilisation})}$.
- **Utilisation Scenarios:**
  - **5.0% Utilisation (Low traffic / idle off-peak):** **0.8537 THB** / 1k predictions ($\$0.0244$)
  - **25.0% Utilisation (Standard production service):** **0.1707 THB** / 1k predictions ($\$0.0049$)
  - **80.0% Utilisation (High sustained production load):** **0.0534 THB** / 1k predictions ($\$0.0015$)

Note for remember: Capacity utilisation is the single most sensitive variable in serving economics; an over-provisioned endpoint with low traffic costs up to $16\times$ more per prediction than an efficiently utilized one.

### 2. Batch Inference Breakeven Analysis
- Keeping a live endpoint warm 24/7 costs 45.36 THB/day in continuous infrastructure spend, meaning you pay for server capacity even when it sits idle between requests. In contrast, a scheduled daily batch job only spins up compute when there is data to process, scores 100,000 predictions in roughly 13 minutes, and immediately shuts down, costing just 0.40 THB/day. Because of this, batch inference is far cheaper whenever traffic is low (under 45 requests/day or 0.0005 RPS) or whenever predictions can be delivered on a delay rather than requiring instant, sub-second responses.


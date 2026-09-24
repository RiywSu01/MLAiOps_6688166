# ITCS355 Lab 3 — Load Test and Serving Report

- **Deployment Platform:** Azure Container Apps (ACA Consumption, East Asia)
- **Endpoint URL:** `https://itcs355-predict.redground-de34b2df.eastasia.azurecontainerapps.io`
- **Model Version:** `2` (Digest: `itcs355-serve@sha256:41e546ed46fe484a05df8c88a49b607b90ebc0136284f336d6e7b180f5584c4c`)
- **Load Test Tool:** Grafana k6 (`loadtest/k6.js`)

---

## 1. Latency Target (Stated Before Measurement)

In accordance with the rubric requirement, the target threshold was committed to git prior to executing the load tests (Git commit `586ef84` in `loadtest/k6.js`):

- **Stated Target:** $p_{95} < 250\text{ ms}$
- **Target Error Rate:** $\text{Failures} < 1\%$ (`rate < 0.01`)

---

## 2. Cold-Start vs. Warm Latency

Because Azure Container Apps is configured with scale-to-zero autoscaling (`minReplicas: 0`) to comply with Azure for Students quota restrictions, cold starts were isolated and measured independently:

| Metric | Measured Duration | Notes |
| :--- | :--- | :--- |
| **Cold-Start Latency** | **12.950 s** | First request after revision scale-to-zero (container allocation + image pull + FastAPI startup + MLflow model unpickle) |
| **Warm Latency** | **0.816 s** | Subsequent request hitting the active replica |
| **Cold-Start Penalty** | **+12.134 s** | Container instantiation overhead |

---

## 3. Concurrency Benchmarks (1, 10, 50 VUs)

All tests were executed for 60 seconds per concurrency level using `make loadtest`.

| Concurrency (VUs) | Throughput (RPS) | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) | Error Rate (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 VU** | 1.62 | 613.7 | 683.6 | 1,050.2 | 1,158.6 | 0.00% (0/97) |
| **10 VUs** | 12.55 | 616.2 | 718.9 | 10,480.0 | 11,672.9 | 0.00% (0/793) |
| **50 VUs** | 70.18 | 581.9 | 678.1 | 8,920.0 | 11,628.9 | 0.00% (0/4,249) |

### Breaking Point Analysis
- **Latency Breaking Point:** The stated target (p95 < 250 ms) is crossed at **1 VU** (p95 = 683.6 ms). 
  - *Root Cause:* The internal server processing time in FastAPI is only $\approx 7\text{ ms}$. The remaining $\sim 600\text{--}680\text{ ms}$ is physical WAN round-trip transit (Thailand client $\leftrightarrow$ Azure Hong Kong datacenter via Envoy ingress). A 250 ms target cannot be achieved over long-haul WAN; achieving p95 < 250 ms requires co-locating the client in the same Azure region (`eastasia`).
- **Throughput / Stability Breaking Point:** Not reached at 50 VUs. Throughput scaled linearly from 1.62 RPS to 70.18 RPS with **0.00% errors** across 5,139 total requests.
- **Autoscaler Impact on p99 / Max:** When concurrency jumped from 1 to 10 VUs and from 10 to 50 VUs, the ACA horizontal pod autoscaler (HPA) spun up additional replicas from 0 to 3. The requests that triggered new container launches absorbed the $\sim 11.6\text{ s}$ cold-start penalty, creating the long tail in p99 and Max while median (p50) remained tight at $\sim 580\text{--}616\text{ ms}$.

---

## 4. Controlled Variable Experiments

### Variable 1: Batch Size (`/predict` vs `/predict/batch`)
- **Experiment:** Compared 100 individual sequential calls to `/predict` against 1 single call with 100 rows to `/predict/batch`.
- **Results:**
  - 100 single calls: **90.21 s** (average $902.1\text{ ms/row}$)
  - 1 batch call (100 rows): **0.81 s** (average $8.1\text{ ms/row}$)
  - **Speedup:** **111.7x faster**
- **Analysis:** Individual requests each pay the WAN network round-trip penalty (100 sequential RTTs). The batch endpoint amortizes network transit over 100 rows in a single TCP packet train and leverages vectorized NumPy/scikit-learn inference, which computes 100 predictions in $\approx 5\text{ ms}$.

### Variable 2: Payload Size & Serialization Dominance
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

### Variable 3: Instance Size Scaling
- **Experiment:** Upgraded the container resource specification from **0.5 vCPU / 1.0 GiB** to **1.0 vCPU / 2.0 GiB**.
- **Results:**
  - Warm Latency on $0.5\text{ vCPU} / 1.0\text{ GiB}$: **0.816 s** ($816.4\text{ ms}$)
  - Warm Latency on $1.0\text{ vCPU} / 2.0\text{ GiB}$: **0.827 s** ($826.7\text{ ms}$)
  - Latency Delta: **$\approx 0\%$ change** (within network jitter noise)
  - Cost Delta: **+100% cost increase** (from $\$0.054/\text{hour} \approx 1.89\text{ THB/hr}$ to $\$0.108/\text{hour} \approx 3.78\text{ THB/hr}$)
- **Analysis:** The machine learning inference model is lightweight ($< 10\text{ ms}$ CPU compute). Because latency is network I/O-bound rather than compute-bound, doubling vCPU and memory doubles infrastructure spend with zero perceptible improvement in end-user latency. Over-provisioning compute is an anti-pattern for this workload.

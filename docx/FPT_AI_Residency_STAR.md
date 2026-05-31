# STAR Interview Answer for FPT AI Residency

**Question:** What is the most challenging research problem you have worked on, and how did you approach it?

**Answer:**

### 1. Situation
While researching Intrusion Detection Systems for IoT networks, I framed the problem as **One-Class Novelty Detection (OCND)**. This approach learns exclusively from benign traffic to detect unseen, zero-day cyber threats, rather than relying on traditional closed-set classifiers.

### 2. Task
The most challenging problem was addressing a major gap in previous works: balancing three conflicting constraints simultaneously:
1. **Resource Constraints:** IoT edge devices require extremely lightweight models.
2. **Training Contamination:** Real-world benign data is rarely 100% clean (label noise).
3. **Diverse Anomaly Structures:** IoT attacks manifest in varied ways (local, global, or clustered).

The core conflict is that detecting diverse anomalies requires tight decision boundaries, which makes the model highly fragile to contamination. Conversely, deep learning solutions that solve both violate IoT resource constraints.

### 3. Action
To resolve this trade-off, I proposed **LOC-NFST** (Local One-Class Null Foley-Sammon Transformation). Instead of using heavy deep learning models, I innovated on linear projections:
* **Pseudo-Class Construction:** I used lightweight K-means clustering to partition the benign data into "pseudo-classes," capturing the multi-modal nature of the traffic.
* **Mathematical Reformulation:** The original NFST algorithm fails in one-class settings (lacking between-class scatter) and in high-sample regimes ($n \gg d$). I solved this by introducing a **spectral relaxation technique using SVD**. Instead of forcing an exact null-space, I projected data into a "near-null" space containing the smallest eigenvalues.
* **Prototype-Based Scoring:** In this projected space, I scored samples based on their distance to the pseudo-class centroids. This mechanism shrinks intra-class variations (neutralizing contamination) while preserving the data's geometry to catch diverse anomalies.

### 4. Result
This approach successfully resolved the trade-offs:
* **Robust Performance:** The model achieved up to **99.39% AUC-ROC** across 4 IoT benchmark datasets (CICIoT2023, ToN-IoT, NBaIoT, BoT-IoT). It remained highly stable across diverse anomaly types and tolerated up to 5% data contamination.
* **Edge-Friendly Efficiency:** By relying on efficient SVD matrix operations with a linear time complexity of **$O(nd^2)$**, the model is extremely lightweight and deployable directly on resource-constrained IoT edge devices.

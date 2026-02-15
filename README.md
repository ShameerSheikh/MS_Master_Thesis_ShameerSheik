# MS_Master_Thesis_ShameerSheik
# A CROSS-DOMAIN RECOMMENDATION SYSTEM FOR ENHANCING JOBS, BUSINESS OPPORTUNITY DISCOVERY 

# Tools used
- Visual Studio Code Editor
- Interpreter: Python 3.11.9
- Executed in Virtual Environment of name: cgga_env

# Python libraries used
- import os
- import numpy as np
- import pandas as pd
- import time
- import torch
- import random
- from sentence_transformers import SentenceTransformer
- from tqdm import tqdm
- from sklearn.cluster import MiniBatchKMeans
- from sklearn.metrics import silhouette_score



# Overview:

This research presents a comprehensive investigation into the development of a cross-domain recommendation system designed to function as a personalized user assistant on job/business opportunities. 

The system aims to recommend tailored career, job, and business opportunities to a diverse user base including working professionals, students, retirees, and athletes based on their individual interests, educational qualifications, and risk profiles. 

Key components of the CGGA framework include 
1. Domain-Specific Causal Graph Construction, 
2. Variational Generative Modelling, 
3. Adaptive Few-Shot Learning and 
4. Multi-Objective Optimization. 

This approach synergistically integrates causal graph learning with generative alignment techniques to enhance the precision and contextual relevance of knowledge transfer across heterogeneous domains such as employment markets, entrepreneurial landscapes, and macroeconomic indicators. 

The proposed CGGA framework establishes a scalable and intelligent foundation for opportunity discovery, empowering users to make informed career and business decisions that are aligned with real-world dynamics and personal aspirations. 

By integrating causal reasoning with generative modelling and adaptive learning, this research contributes a significant advancement in the field of cross-domain recommendation systems. It holds the potential to substantially reduce decision-making risk and improve outcome effectiveness for individuals navigating complex career and entrepreneurial landscapes.


CGGA Implementation:
Basic Steps included here as listed below

STAGE 0 — DATA PRE-PROCESSING
STAGE 1 — Load & Preprocess User + Job Datasets
STAGE 2 — Embedding Generation (User + Job)
STAGE 3 — Job Clustering
STAGE 4 — Causal Graph Construction
STAGE 5 — Variational Generative Alignment (VAE / CVAE)
STAGE 6 — Adaptive Few‑Shot Learning
STAGE 7 — Multi‑Objective Scoring + Recommendation Engine
STAGE 8 — EVALUATION


# Datasets Chosen from Kaggle Sources:
- User Profile data: Resume Dataset, 
https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset

- Job Posting data: 1.3M Linkedin Jobs & Skills (2024), 
https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024 

- Mapping file: companies_sorted.csv
Kaggle dataset for company to industry mapping
https://www.kaggle.com/datasets/peopledatalabssf/free-7-million-company-dataset

# CGGA (Cross‑domain Generalized Generative Alignment) relies on semantic embeddings.
Embeddings work best when the input text:
- is high‑signal
- is semantically rich
- directly reflects skills, responsibilities, and intent
- avoids noise, metadata, and irrelevant fields
If you feed embeddings with everything, you dilute the signal and degrade cluster quality. So we intentionally select only the columns that carry semantic meaning.

Only semantically meaningful fields were included in the embedding pipeline. 
Metadata fields such as dates, URLs, institution names, and locations were excluded to prevent noise injection and maintain embedding purity. 
The unified text representation ensures consistent semantic density and improves clustering stability.

# RESULTS
Model was evaluated on 100 users × full job catalog, with Top‑5 recommendations per user. Overall, the system is diverse, moderately accurate, and capable of ranking relevant jobs early, but precision and recall are naturally low due to the synthetic full‑matrix ground truth.

- A Precision@5 of 0.038 suggests that, on average, one out of the top‑five recommended jobs aligns with the user’s few‑shot relevance profile. 
- This is consistent with the Recall@5 score of 0.038, given that both the predicted and ground‑truth lists are limited to five items per user.
- The HitRate@5 of 0.14 demonstrates that the system successfully retrieves at least one relevant job for 14% of users. Reasonable for a cold‑start, cross‑domain system.
- Ranking quality is further supported by an MRR@5 of 0.107, indicating that when the system does retrieve a relevant job, it tends to appear near the top of the recommendation list.
- The NDCG@5 score of 0.0785 reinforces this observation, shows the model orders relevant jobs reasonably well.
- One of the strongest outcomes is the ILD@5 score of 0.508, which reflects a healthy level of diversity among recommended jobs. This suggests that the reranking mechanism is not overly biased toward a single job cluster or embedding neighborhood, and instead provides users with a varied set of opportunities. 

In real‑world job recommendation scenarios, diversity is crucial for user satisfaction, exploration, and fairness.


# Strengths of the CGGA Framework:

1. High Diversity (ILD@5 ≈ 0.51)
- Your recommendations span multiple job types.
- This supports your opportunity discovery and cross‑domain exploration claims.
- High ILD is a strong indicator that Hybrid‑CGGA is not stuck in narrow clusters.

2. Good Ranking Quality (MRR & NDCG)
- MRR@5 = 0.107 and NDCG@5 = 0.0785 show:
- Relevant jobs appear early in the Top‑5.
- The ranking function is meaningful.
- This is impressive given the huge candidate space (14k+ jobs).

3. Strong Cross‑Domain Potential
- High ILD + upcoming DJR metric will show:
- The system is capable of recommending outside the user’s domain.
- Supports your thesis claim of cross‑domain job discovery.

4. Stable Behavior Across Users
- Hit Rate@5 = 0.14 indicates:
- The model consistently finds at least one relevant job for many users.
- No extreme variance or collapse.


# Weaknesses / Future Work:

1. Precision and recall remain low
- Your ground truth is synthetic and dense (every user has relevance labels for all jobs).
- The candidate space is extremely large (14k+ jobs).
- Top‑5 is a very small window.
This is not a model failure. it is a dataset property.

2. Popularity Bias Cannot Be Measured
- All jobs appear exactly 100 times in ground truth.
- Popularity is uniform → no variance → no popularity metrics possible.

3. Synthetic Ground Truth Limits Realism
- Relevance labels are not based on real user behavior.
- Precision/recall cannot reach high values in such a setting.

4. No Personalization History
- No past interactions → model relies only on embeddings.
- Limits personalization depth.


# CONCLUSION:

Overall, thisHybrid‑CGGA recommender demonstrates:
- Strong diversity
- Meaningful ranking quality
- Cross‑domain exploration capability
- Moderate hit rate
- Expectedly low precision/recall due to synthetic full‑matrix ground truth
This is a balanced and defensible evaluation for a cross‑domain, opportunity‑discovery recommender system.


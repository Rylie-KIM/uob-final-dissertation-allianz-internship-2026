# Foundational Literature for the Allianz Internship Project

*Topic: Identifying and Mitigating Self-Fulfilling Prophecy Loops in Machine Learning · Interview: Wed 13 May 2026, 3pm UK*

These are the **two papers most directly aligned** with the dissertation's framing. Together they cover the *evaluation problem* (how do you measure performance when labels are selected by past decisions?) and the *dynamics problem* (why do feedback loops compound, and how do you break them?). If you read nothing else before the interview, read these.

---

## 📄 Paper 1 — *The Selective Labels Problem* (Lakkaraju et al., KDD 2017) 🎯

| Field | Detail |
|---|---|
| **Full citation** | Lakkaraju, H., Kleinberg, J., Leskovec, J., Ludwig, J., & Mullainathan, S. (2017). *The Selective Labels Problem: Evaluating Algorithmic Predictions in the Presence of Unobservables*. KDD '17, pp. 275–284. |
| **Venue** | 23rd ACM SIGKDD Conference on Knowledge Discovery and Data Mining |
| **Why it's #1 for this project** | Mathematically frames the *exact* problem in the Allianz brief: outcome labels (e.g. fraud confirmed) only exist for cases a human decision-maker chose to investigate. The paper's running examples include **insurance** alongside criminal justice and healthcare. |
| **Core idea — "contraction"** | Exploit *heterogeneity across decision-makers* (some judges/investigators are stricter than others) to recover a comparison of algorithm vs human performance — without requiring counterfactual inference and without assuming all confounders are observed. |
| **What to extract** | (i) The selective-labels formal setup (notation, observed-vs-unobserved labels). (ii) Why standard metrics break under this regime. (iii) The contraction technique as a *baseline* you'd compare against IPS / RDD in the dissertation. |
| **What's *not* in it** | The dynamics across model versions — Lakkaraju treats one decision-maker at a time. Combine with Paper #2 for the temporal/feedback dimension. |

**Links (verified):**
- ACM Digital Library (canonical): <https://dl.acm.org/doi/10.1145/3097983.3098066>
- Stanford open-access PDF: <https://cs.stanford.edu/~jure/pubs/contraction-kdd17.pdf>
- PMC open-access mirror: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5958915/>

**Reading priority for interview prep (in order):**
1. Abstract + Section 1 (Introduction) — 10 min
2. Section 2 (Selective Labels Problem) — formal setup, 15 min
3. Section 4 (Contraction technique) — high-level only, 10 min
4. Skip the empirical sections unless time permits

---

## 📄 Paper 2 — *Runaway Feedback Loops in Predictive Policing* (Ensign et al., FAccT 2018)

| Field | Detail |
|---|---|
| **Full citation** | Ensign, D., Friedler, S. A., Neville, S., Scheidegger, C., & Venkatasubramanian, S. (2018). *Runaway Feedback Loops in Predictive Policing*. Proceedings of the 1st Conference on Fairness, Accountability and Transparency (FAccT), PMLR 81:160–171. |
| **Venue** | 1st ACM Conference on Fairness, Accountability and Transparency (FAccT/FAT*) |
| **Why it's #2 for this project** | The canonical paper that **mathematically proves** why action-driven feedback loops compound, using a Pólya urn process. Predictive policing is structurally isomorphic to insurance fraud investigation — same "model decides where to look → labels only emerge from where it looked → next model is biased toward the same places." |
| **Core idea** | (i) Model the loop as a Pólya urn — show that under naive retraining, observed crime/fraud rates diverge from true rates. (ii) Propose a **black-box intervention** that re-injects unbiased samples to break the loop and recover true rates. |
| **What to extract** | (i) The structural argument for *why* ε-greedy / random exploration works (this is your Phase 5 motivation). (ii) The fairness framing — feedback loops disproportionately harm under-investigated segments (directly transferable to UK Equality Act / Consumer Duty argument). |
| **What's *not* in it** | Causal-inference heavy machinery (no DoWhy, no formal DAG identification). Combine with the broader causal-ML literature for that. |

**Links (verified):**
- PMLR (canonical): <https://proceedings.mlr.press/v81/ensign18a.html>
- arXiv preprint: <https://arxiv.org/abs/1706.09847>
- Friedler-hosted PDF: <https://friedler.net/papers/feedbackloops_fat18.pdf>

**Reading priority for interview prep (in order):**
1. Abstract + Introduction — 10 min
2. Section 2 (Model setup) — Pólya urn formulation, 15 min
3. Section 4 (Interventions) — what fixes the loop, 10 min
4. Skim the experiments to confirm the mechanism

---

## How the Two Papers Fit Together

| Dimension | Lakkaraju (2017) | Ensign (2018) |
|---|---|---|
| **Question they answer** | "How do I *evaluate* a model when labels are selected?" | "Why does the loop *worsen* over time, and how do I break it?" |
| **Time scope** | Single decision point (cross-sectional) | Multi-period dynamics (temporal) |
| **Mathematical tool** | Contraction via decision-maker heterogeneity | Pólya urn process |
| **Maps to dissertation phase** | Phase 3 (Unbiased Evaluation) | Phase 5 (Randomisation Strategy) |
| **Intervention they propose** | Compare contraction-recovered metrics | ε-greedy / sample re-injection |
| **Domain examples** | Bail, healthcare, **insurance** | Predictive policing |

Together they cover **detection (Lakkaraju)** and **mitigation (Ensign)** — the two halves of the project brief.

---

## Interview Crosswalk — Which Paper to Cite for Which Question

| Likely interview prompt | Cite | One-line response |
|---|---|---|
| *"Why is standard ML evaluation insufficient here?"* | **Lakkaraju 2017** | "The selective-labels framing in Lakkaraju et al. shows that when labels are observed only on selected cases, standard precision/recall don't generalise to the population." |
| *"Why doesn't naive retraining solve this?"* | **Ensign 2018** | "Ensign et al. prove via a Pólya urn model that naive retraining can run away — observed rates diverge from true rates." |
| *"What prior work informs your approach?"* | **Both, as a pair** | "Lakkaraju addresses the evaluation problem cross-sectionally; Ensign addresses the dynamics. My project sits at their intersection — using Allianz's multi-version dataset as a quasi-experiment." |
| *"How does your work differ from the existing literature?"* | **Both** | "Lakkaraju assumes multiple decision-makers with varying strictness; in insurance we have *multiple model versions* deployed sequentially, which gives a similar quasi-experimental structure but along the time axis. Ensign proves the loop in a stylised model; we'd test it on real claims data." |
| *"Why is this fairness-relevant?"* | **Ensign 2018** | "Feedback loops disproportionately harm under-investigated segments — under UK Consumer Duty / Equality Act 2010, that's a regulatory exposure, not just a technical one." |

---

## Suggested Pre-Interview Reading Plan (≈90 min total)

| Block | Time | Task |
|---|---|---|
| 1 | 25 min | Lakkaraju — abstract, intro, Section 2 (selective-labels setup) |
| 2 | 15 min | Lakkaraju — skim Section 4 (contraction) for high-level intuition |
| 3 | 25 min | Ensign — abstract, intro, Section 2 (Pólya urn model) |
| 4 | 15 min | Ensign — Section 4 (interventions) |
| 5 | 10 min | Re-read the *Crosswalk* table above and rehearse 1-sentence pitches |

> **Goal of this reading is not memorisation** — it's being able to (a) name the paper, (b) state its core idea in one sentence, and (c) connect it to your project's framing, all in under 30 seconds when prompted.

---

*Last updated: 2026-05-10. All links verified via web search before saving.*

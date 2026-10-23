# OPTION A: Gated, Domain-Transferable Uncertainty Steering

**Living project document. This file is the source of truth for Option A.**

| Field | Value |
|---|---|
| Owner | [Name, ID], part-time pre-PhD, BITS Pilani Goa Campus |
| Deadline | 28 Nov 2026 (project report) |
| Created | 21 Oct 2026 |
| Last updated | 21 Oct 2026 |
| Current step | Step 1 (in progress) |
| Overall status | Planning |

---

## 0. How to use this document

1. Read this file at the start of every work session and update it at the end.
2. Never delete a past decision. If a decision changes, add a new row to the Decisions Log that supersedes the old one.
3. Every result goes in the Experiment Log with its date, config, and outcome, including failures and null results.
4. Mark anything not yet confirmed as **[UNVERIFIED]**. Move it to a confirmed statement only after checking the source or running the experiment.
5. Add one line to the Changelog (Section 16) for every edit.
6. Keep facts (what the papers or experiments show) separate from assumptions and plans.

---

## 1. One-line summary

Reimplement USteer (a training-free, uncertainty-guided steering method), test whether its small gains survive proper statistics, add a confidence-gated variant that steers only when the model is unsure, and test whether a steering strength tuned on general reasoning data transfers to medical question answering.

---

## 2. Scope and context

- **Broad area:** Reliability of large language model (LLM) reasoning; inference-time methods that reduce hallucination without retraining.
- **Specific task:** Use a small open model's own confidence signal to steer hidden states during decoding, improving answers on multi-step reasoning, first on general question answering and then on medical question answering.
- **Current state (from the USteer paper, Liu et al., 2026, arXiv 2609.38962):**
  - Consistent but small gains on CommonsenseQA, StrategyQA, GSM8K, AQuA across Llama-3.2-1B, Qwen-2.5-3B, Qwen-2.5-14B, Qwen3-32B.
  - Llama-3.2-1B: average gain +2.6 points; bottom-25% confidence examples +6.7; bottom-5% +14.4.
  - Beats or matches DoLa, ITI, TruthX in most cells, with latency close to standard decoding.
  - Gains shrink with scale (Qwen3-32B: about +0.4 to +0.5, none on AQuA).
- **The gap (visible in or around the paper):**
  1. Appendix C states the 95% bootstrap intervals are wide relative to differences between methods, so gains are not statistically conclusive under greedy decoding.
  2. Section 4.2 notes steering toward confidence can reinforce confidently wrong answers. Strength is tuned per dataset.
  3. Only short answers (about 35 tokens on average) on general-domain data. No medical evaluation. Code not released.
  4. USteer steers at every decoding step. A preliminary search (21 Oct 2026) found no training-free method gated by its own confidence score. **[Step 1 update, 21 Oct 2026: two gated steering papers read in full (arXiv:2608.23666, arXiv:2609.01878 GAPS); neither gates on the model's own answer-confidence on reasoning/QA. See Section 4.4 and D-007/D-008. Broader search still ongoing — not yet exhaustive.]**

---

## 3. Research questions and success criteria

| ID | Question | Confirmed if | Status |
|---|---|---|---|
| RQ1 | Do USteer's gains reproduce under proper statistics (paired tests, intervals, multiple seeds under sampling)? | Gain over standard decoding is significant (McNemar, paired bootstrap) on at least one model and dataset. A non-significant result is also a valid finding. | Open |
| RQ2 | Does confidence gating beat plain USteer, especially on confidently wrong answers? | Gated USteer matches or beats plain USteer in net accuracy and flips fewer correct answers to wrong. | Open |
| RQ3 | Does a strength tuned on general data transfer to medical QA? | Compare transferred versus re-tuned strength on MedQA and PubMedQA. A gap means transfer fails. | Open |
| RQ4 (stretch) | How does steering compare with or combine with RAG or LoRA fine-tuning? | Only attempted if Steps 1 to 7 finish early. | Deferred |

---

## 4. Method specification

### 4.1 USteer mechanics (from the paper)

- **Logit lens.** For layer output x^l, normalise with the model's final RMSNorm: z^l = D x^l / RMS(x^l), where D is the diagonal learned gain. Predicted token at layer l: ĉ^l = argmax_v (w_v^T z^l), where w_v are unembedding vectors.
- **Confidence guidance score (Definition 3.1):** g(z) = (w_ĉ^T z)^2 / ||z||^2. The paper calls this a squared cosine similarity. **[VERIFY: as printed the formula omits ||w_ĉ||^2, so check whether the paper's code or text normalises w_ĉ before implementing.]**
- **Gradient (Lemma 3.2, Eq. 6):** ∇_z g = (2 (w_ĉ^T z) / ||z||^2) · (w_ĉ − ((w_ĉ^T z) / ||z||^2) z).
- **Chain rule through RMSNorm (Eq. 7):** ∇_x g = (1/||x||) (I − x x^T / ||x||^2) D ∇_z g.
- **Update (Eq. 8):** x ← x + α · ∇_x g / ||∇_x g||. Single step, applied at selected upper intermediate layers, at every decoding step.
- No backpropagation through the network is needed. Forward hooks plus the closed-form gradient suffice.
- **Clarification on α (21 Oct 2026, from Step 1 discussion):** ∇_x g only determines the *direction* of the push; dividing by its own norm discards all magnitude information from the gradient. The step size is therefore entirely α — a single scalar fixed in advance by grid search per (model, dataset) pair (Section 4.2), applied identically at every token and every steered layer, regardless of the model's current confidence g. Plain USteer has no mechanism to scale down or skip the push when the model is already confident (right or wrong) at that step — the gradient supplies the heading, α is the only throttle, and it never moves during generation. This is the precise structural gap gated USteer (Section 4.4) closes.

### 4.2 Paper hyperparameters (reproduction targets)

- Steered layers: last 3 intermediate layers for Llama-3.2-1B; last 8 for Qwen-2.5-3B.
- Strength sweep: α in {0.1, ..., 0.9}. Paper's chosen values: Llama-1B: CSQA 0.2, StrategyQA 0.1, GSM8K 0.3, AQuA 0.03. Qwen-3B: CSQA 0.9, StrategyQA 0.3, GSM8K 0.5, AQuA 0.3.
- Decoding: greedy for main results; sampling at T in {0.2, 0.5, 0.8, 1.0} with five seeds.
- Prompting: few-shot chain-of-thought from Wei et al. (2022). Answer extracted after "So the answer is" (CSQA, StrategyQA) or "The answer is" (GSM8K, AQuA).

### 4.3 Reproduction targets (from the paper, Table 1 and Table 8)

| Model | Dataset | Standard | USteer | 95% CI half-width |
|---|---|---|---|---|
| Llama-3.2-1B | CSQA (n=1221) | 50.94 | 54.05 | ±2.78 |
| Llama-3.2-1B | GSM8K (n=1319) | 34.19 | 36.24 | ±2.58 |
| Qwen-2.5-3B | CSQA | 77.72 | 79.20 | ±2.29 |
| Qwen-2.5-3B | GSM8K | 79.91 | 80.97 | ±2.12 to 2.16 |

### 4.4 Proposed gated variants (our contribution, to be designed)

- **G1, hard gate:** steer only at decoding steps where confidence is below threshold τ.
- **G2, soft gate:** scale α by a decreasing function of confidence.
- **Design notes:**
  - Raw g values are small (about 0.02 to 0.08 in the paper's Figure 2), so τ and any scaling must be set from the validation distribution (for example percentiles), not as absolute constants.
  - **Open design question:** gate on the layer-wise g under the logit lens, or on the final-layer softmax confidence? Decide in Step 6 and log the decision.
  - Gating granularity: per decoding step (token level). Per-sequence gating is a possible ablation.
  - Restated motivation (see Section 4.1 clarification): plain USteer's step size is a fixed α with zero dependence on per-step confidence g. Gating introduces exactly that missing dependence — nothing more, nothing less.
- **Closest related work on gating (read in full, Step 1, 21 Oct 2026):**
  - **arXiv 2608.23666** (Tripathi, Neupane, Mitra, Mittal, Amiri Golilarz, Rahimi — Univ. of Alabama, submitted 24 Aug 2026) — gated activation steering for medical QA via ITI. Steering directions for hallucination and sycophancy are learned offline from contrastive clinical pairs and applied to causally-verified attention heads; gates are behavior-specific detectors (triggered by detected hallucination/sycophancy patterns), not the model's own uncertainty. Tested on a 4B model vs. 100B+ models on EHR-grounded clinical QA; gated steering preserved correct answers under user pressure in 551/600 trajectories vs. 30/600 unsteered. **Classification: related work, not a baseline** — different task (dialogue robustness under pressure, not single-turn QA accuracy) and different method family (ITI, not logit-lens steering).
  - **arXiv 2609.01878, GAPS** (Fereidouni, Haider, Sajjad, Siddique, submitted 1 Sep 2026) — newly surfaced during this search, not previously in scope. Genuinely training-free dimension-level gating for activation steering: a separability gate (AUROC) and a posterior gate (Gaussian model) decide which neurons and when to steer, no learned classifier. Tested on Gemma-3-4B and Qwen-3-1.7B for toxicity mitigation and concept removal (RealToxicityPrompts, OneSeC); builds on CAST/DSAS; does not cite or compare to USteer. **Classification: related work only** — training-free and gated like our proposal, but the gate signal is concept-separability statistics for a predefined target concept, not the model's own confidence about its current generated answer, and the task is toxicity/concept removal, not reasoning QA.
  - **Net effect on novelty (D-002):** holds after this round. No paper found yet that gates training-free steering by the model's own confidence score on reasoning/QA tasks. GAPS is the closest in spirit (training-free + gated) and should be cited explicitly to pre-empt a reviewer raising it.

---

## 5. Data and evaluation protocol

### 5.1 Datasets

| Dataset | Role | Notes |
|---|---|---|
| GSM8K (test, 1,319) | Reproduction and control | Free-form numeric answers. |
| CommonsenseQA (validation, 1,221) | Reproduction and control | Multiple choice. |
| MedQA | Medical transfer | Risk: small models may score near chance. Check baseline first. **[UNVERIFIED: size and licence.]** |
| PubMedQA (labeled) | Medical transfer | Yes/no/maybe, imbalanced. Report macro-F1 or balanced accuracy. **[UNVERIFIED: size and licence.]** |
| MedMCQA (optional) | Fallback medical set | Large; Indian medical entrance exams. **[UNVERIFIED: licence.]** |

Avoid relying on StrategyQA (229) and AQuA (254): intervals of about ±6 points.

### 5.2 Metrics

- Accuracy with bootstrap 95% confidence intervals.
- McNemar's test on paired predictions (standard versus steered).
- Flip counts: correct to wrong, wrong to correct.
- Accuracy by confidence quantile (bottom 5%, 25%, all), as in the paper's Table 2.
- Latency (ms per token).
- For PubMedQA: macro-F1 or balanced accuracy.

### 5.3 Models

- Llama-3.2-1B-Instruct (requires accepting the licence on Hugging Face).
- Qwen-2.5-3B-Instruct.
- A larger model (7B to 8B) only as a later extension if compute allows.

---

## 6. Step plan with status

Status key: [ ] not started, [~] in progress, [x] done, [!] blocked.

| # | Step | Target dates | Done when | Status |
|---|---|---|---|---|
| 1 | Check novelty and settle scope | 21 to 22 Oct | Fuller search for gated or adaptive steering is done. 2608.23666 is read and classified as baseline or related work only. Decision logged. | [~] |
| 2 | Environment and baselines | 22 to 27 Oct | Colab set up; models load; few-shot CoT prompts and answer extractors for GSM8K and CSQA work; standard-decoding results land near the targets in 4.3. | [ ] |
| 3 | Implement steering | 27 Oct to 1 Nov | Logit lens, g(z), closed-form gradient, hooks done. Gradient matches autograd on a small tensor. g separates correct from incorrect (AUROC check, paper Appendix F). | [ ] |
| 4 | Reproduce USteer | 1 to 5 Nov | Strength and layer sweep on a validation split; full runs on both models; compared with 4.3. Mismatch is documented, not hidden. | [ ] |
| 5 | Proof-of-gap experiments | 4 to 10 Nov | Paired tests and intervals done; confident-error analysis done; medical baselines run and confirmed above chance; medical prompts and extractor work. | [ ] |
| 6 | Gated USteer | 8 to 14 Nov | G1 and G2 implemented; thresholds tuned on validation only; compared with plain USteer and baseline, including flip counts. | [ ] |
| 7 | Medical transfer | 11 to 17 Nov | Transferred versus re-tuned strength evaluated for plain and gated steering on MedQA and PubMedQA. | [ ] |
| 8 | Analysis and write-up | 15 to 28 Nov | Final tables with intervals, ablations, report written, code in a repository. | [ ] |

Schedule is tentative and shared with Option B (see Section 10). Update dates whenever they slip.

---

## 7. Experiment log

One row per run. Never overwrite. Record failures.

| ID | Date | Step | Model | Dataset | Method and config (α, layers, τ, decoding, seed) | Result | Notes and next action |
|---|---|---|---|---|---|---|---|
| E-001 | | | | | | | |

---

## 8. Results (fill as experiments complete)

### 8.1 Reproduction

| Model | Dataset | Standard (ours) | Paper standard | USteer (ours) | Paper USteer | Significant? |
|---|---|---|---|---|---|---|
| | | | | | | |

### 8.2 Gated versus plain steering

| Model | Dataset | Standard | USteer | G1 | G2 | Flips c→w (USteer / G1 / G2) | Flips w→c |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

### 8.3 Medical transfer

| Model | Dataset | Standard | α from general data | α re-tuned on medical | Gap |
|---|---|---|---|---|---|
| | | | | | |

### 8.4 Accuracy by confidence quantile

| Method | All | Bottom 25% | Bottom 5% |
|---|---|---|---|
| | | | |

---

## 9. Decisions log

| ID | Date | Decision | Reason | Supersedes |
|---|---|---|---|---|
| D-001 | 21 Oct 2026 | Pursue Option A (gated USteer) alongside Option B. | Fits professor's baseline-comparison style; feasible on Colab; clear gap. | |
| D-002 | 21 Oct 2026 | Frame novelty as a training-free, self-confidence-gated variant, not generic gating. | Gating exists in other work (2608.23666 and vision-language papers). | |
| D-003 | 21 Oct 2026 | Medical transfer is one of three contributions, not the main one. | On its own it is a thin project. | |
| D-004 | 21 Oct 2026 | Use Llama-3.2-1B and Qwen-2.5-3B first. | Paper's main models; fit a single Colab GPU. | |
| D-005 | 21 Oct 2026 | RAG and LoRA comparison deferred to stretch. | Time: about five weeks and two projects. | |
| D-006 | 21 Oct 2026 | Gate-signal choice (layer-wise g versus softmax) left open until Step 6. | Needs data to decide. | |
| D-007 | 21 Oct 2026 | Read arXiv:2608.23666 in full; classified as related work, not a baseline. | Gates are behavior-specific detectors on learned ITI directions, not self-confidence gating; different task (dialogue pressure robustness) from our QA accuracy focus. | |
| D-008 | 21 Oct 2026 | Found and added arXiv:2609.01878 (GAPS) as related work; novelty claim (D-002) still holds. | GAPS is training-free and gated but gates on concept-separability statistics for a fixed target concept, not on the model's own answer confidence; different domain (toxicity/concept removal vs. reasoning QA); must cite to pre-empt a reviewer. | |

---

## 10. Cross-project scheduling

- Both Option A and Option B run in parallel. Option A is the primary project; Option B is limited to its proof-of-gap pilot unless Option A finishes early.
- Option B's steps 1 to 3 are pure coding with no API cost and can fill gaps while Colab jobs run.
- Colab sessions disconnect: checkpoint outputs to Google Drive after every batch and make scripts resumable.

---

## 11. Open questions

- [ ] Has gated or adaptive steering using a model's own confidence been published? Partial answer (21 Oct 2026): two gated steering papers read (2608.23666, GAPS 2609.01878); neither gates on the model's own answer confidence on reasoning/QA. Novelty holds so far; broader search not yet exhaustive. (Step 1.)
- [ ] Does the gradient formula need ||w_ĉ|| normalisation? (Step 3.)
- [ ] Are MedQA, PubMedQA, MedMCQA sizes and licences as assumed? (Step 5.)
- [ ] Will Llama-3.2-1B score above chance on MedQA? (Step 5.)
- [ ] Gate on layer-wise g or final-layer softmax? (Step 6.)
- [ ] Does the professor expect a RAG or fine-tuning comparison? (Ask when sharing the proposal.)
- [ ] Which Colab GPU is available in practice, and what is steered-run throughput? (Step 2.)

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Gains fall within noise | Larger evaluation sets, paired tests, multiple seeds under sampling. A null result is still reportable. |
| Small models near chance on medical sets | Check baselines in Step 5; switch to MedMCQA or a larger model. |
| Medical sets test knowledge, not reasoning | State this limit; interpret null results carefully. |
| Cannot reproduce paper numbers | Document the mismatch; check hook placement, normalisation, and prompt format first. |
| Competing gated or adaptive work appears | Early literature check; frame novelty narrowly. |
| Colab disconnects or GPU unavailable | Checkpointing; pay-as-you-go fallback. |
| Time: about five weeks, two projects | Option A first; Option B limited to its pilot. |

---

## 13. Resources and environment

- Compute: Google Colab Pro; pay-as-you-go credit as fallback. No local GPU.
- Code: no official USteer code released; implement from paper Eqs. 2 to 8.
- Access: Hugging Face account and Llama licence acceptance needed.
- Storage: Google Drive for checkpoints and results.
- Repository: [to be created in Step 2].

---

## 14. Related work and references

- Liu, L., Hou, Q., Jian, Y., Pourreza, R., Ghavamzadeh, M., Memisevic, R., Qin, Y., Cai, H. Alleviating Hallucination in Reasoning Tasks with Training-Free Uncertainty-Guided Steering. arXiv:2609.38962, 2026. **(Core paper, read in full.)**
- Tripathi, H., Neupane, S., Mitra, S., Mittal, S., Amiri Golilarz, N., Rahimi, S. Gated Activation Steering for Reducing Sycophancy and Hallucination in Medical Question Answering. arXiv:2608.23666, 2026. **(Read in full 21 Oct 2026; related work, not a baseline — see Section 4.4 and D-007.)**
- Fereidouni, M., Haider, M. U., Sajjad, H., Siddique, A. B. GAPS: Dimension-Level Gates for Conditional Activation Steering. arXiv:2609.01878, 2026. **(Found 21 Oct 2026; training-free and gated, but different gate signal and domain — see Section 4.4 and D-008.)**
- Chapra, S., Kumar, D., Mandal, M., Sinha, Y. The Complexity Ceiling Benchmark. CTB@ICML 2026. **(Used for Option B; possible later extension for Option A.)**
- Chang, T. A., Arnett, C., et al. Global PIQA. arXiv:2510.24081. **(Not central; possible later multilingual robustness check.)**
- Baselines named in the USteer paper: DoLa (Chuang et al.), ITI (Li et al.), TruthX (Zhang et al.).

---

## 15. Publication path

- Target: workshop paper on reliability or trustworthy NLP or clinical NLP after the report. **[Check specific deadlines yourself.]**
- Needed for a stronger submission: larger model (7B to 8B), more datasets, released code, and a fuller novelty check.
- Check dataset licences before releasing any derived artifacts.

---

## 16. Changelog

| Date | Change |
|---|---|
| 21 Oct 2026 | Document created from planning discussion. Status: planning. |
| 21 Oct 2026 | Step 1: read arXiv:2608.23666 in full, classified as related work (D-007). Found and read arXiv:2609.01878 (GAPS), added as related work (D-008). Updated Sections 2, 4.4, 9, 11, 14. Novelty claim D-002 holds so far; broader literature search still open. |
| 23 Oct 2026 | Clarified in Section 4.1/4.4 that USteer's α is a fixed scalar (no per-step dependence on confidence g) — gradient gives direction only, magnitude is discarded by normalization. Sharpens the stated motivation for gated USteer. |

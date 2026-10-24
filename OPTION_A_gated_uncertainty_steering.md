# OPTION A: Gated, Domain-Transferable Uncertainty Steering

**Living project document. This file is the source of truth for Option A.**

| Field | Value |
|---|---|
| Owner | [Name, ID], part-time pre-PhD, BITS Pilani Goa Campus |
| Deadline | 28 Nov 2026 (project report) |
| Created | 21 Oct 2026 |
| Last updated | 24 Oct 2026 |
| Current step | Step 1 done; Step 2 (not started) |
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
  1. Appendix C states the 95% bootstrap intervals are wide relative to differences between methods, so gains are not statistically conclusive under greedy decoding. **[Confirmed 23 Oct 2026, full text: CIs are about ±6 points for StrategyQA/AQuA, ±3 or less for CSQA/GSM8K; paper's own words: "precluding conclusive claims of statistical significance under greedy decoding."]** The paper separately reports (Section 4.5) that under stochastic decoding, confidence intervals are mostly non-overlapping, and it calls those gains statistically significant. **RQ1 scope note:** our reproduction should target greedy decoding specifically (where the paper itself admits the gap) and should also run a paired McNemar test, which the paper does not report and which is more appropriate than independent bootstrap CIs for same-example paired comparisons.
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
- **Confidence guidance score (Definition 3.1):** g(z) = (w_ĉ^T z)^2 / ||z||^2. The paper calls this a squared cosine similarity. **[VERIFIED 23 Oct 2026, full text checked: the formula as printed in the paper itself has no ||w_ĉ||^2 term and w_ĉ is not normalised anywhere in Def. 3.1 or the Lemma 3.2 gradient — this is intrinsic to the paper, not a transcription error on our side. "Squared cosine similarity" is a loose description, not literally true unless all w_v happen to be near unit-norm (unembedding rows are not normalised in standard training). Implement exactly as printed for reproduction fidelity; flag the discrepancy explicitly in the write-up.]**
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
  - **Candidate protocol for Step 6 (not yet run, not a decision):** (1) log g (and softmax max-prob) over a validation split, unsteered; (2) bucket by percentile and check where accuracy actually drops, rather than picking a round-number threshold; (3) set τ at that data-justified percentile of this model's own distribution on this dataset, not an absolute constant; (4) for G2, fit a smooth decreasing function (e.g. sigmoid) on the same validation data instead of a hard cutoff; (5) decide g(z) vs. softmax as the gate signal by comparing their AUROC against correctness on validation data (paper does this for g(z) alone in Appendix F). Keep this validation split disjoint from whatever test accuracy gets reported, per the leakage lesson from D-012/D-013.
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
| 1 | Check novelty and settle scope | 21 to 24 Oct | Fuller search for gated or adaptive steering is done. 2608.23666 is read and classified as baseline or related work only. Decision logged. | [x] |
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
| D-009 | 23 Oct 2026 | Confirmed from full-text read of arXiv:2609.38962 that the g(z) formula's missing ||w_ĉ||^2 term is intrinsic to the paper, not our transcription error; implement as printed. Scoped RQ1 to greedy decoding plus a McNemar test, after finding the paper claims significance under stochastic decoding (Section 4.5) but not greedy (Appendix C). | Resolves the prior [VERIFY] flag; sharpens RQ1 so we're not "reproducing" a gap the paper already partially closed under sampling. | |
| D-010 | 23 Oct 2026 | Re-verified D-009's two claims against raw paper HTML directly (no AI-summarizer in between), to avoid building claims on a WebFetch paraphrase. Greedy-vs-stochastic significance gap is openly disclosed by the authors (including in their own NeurIPS checklist) — not claimable as "our discovery," only as a sharper follow-up test (McNemar). Validation/test overlap for α-tuning is now confirmed verbatim, not just suspected — framed as a subfield convention weakness, not misconduct. | Distinguishes "disclosed limitation we extend with a better test" from "undisclosed flaw we exposed," so RQ1's framing in the final report doesn't overclaim. | |
| D-011 | 23 Oct 2026 | Confirmed DoLa, ITI, TruthX descriptions against their own original papers, independent of USteer's characterization. All three predate USteer and are not novelty threats. | Verifies D-002's framing ("trained, static direction" vs. our training-free, dynamic, confidence-gated approach) rather than just assuming USteer described its own baselines fairly. | |
| D-012 | 23 Oct 2026 | Found two unread candidate papers (arXiv:2603.13366 "Latent Entropy-Aware Decoding", arXiv:2511.10292 "Adaptive Residual-Update Steering") that may be closer novelty threats than anything read so far. Step 1's "fuller search" criterion stays open until both are read and classified. | Title-matching alone has been unreliable this round (GAPS looked close by name but gated on something else) — a real read is required before either confirming or ruling out overlap. | |
| D-013 | 24 Oct 2026 | Read both arXiv:2603.13366 (LEAD) and arXiv:2511.10292 (RUDDER) from raw source text; classified both as related work, not novelty threats. Novelty claim (D-002) holds after four related papers checked. Step 1's "fuller search" criterion is now considered satisfied. | Both are training-free and adaptively gated (closest architectural analogs found), but neither gates on the model's own answer-confidence, and neither does gradient-derived activation steering on text reasoning/QA. | |

---

## 10. Cross-project scheduling

- Both Option A and Option B run in parallel. Option A is the primary project; Option B is limited to its proof-of-gap pilot unless Option A finishes early.
- Option B's steps 1 to 3 are pure coding with no API cost and can fill gaps while Colab jobs run.
- Colab sessions disconnect: checkpoint outputs to Google Drive after every batch and make scripts resumable.

---

## 11. Open questions

- [x] Has gated or adaptive steering using a model's own confidence been published? **Answer as of 24 Oct 2026, after checking four related papers (2608.23666, GAPS 2609.01878, LEAD 2603.13366, RUDDER 2511.10292): not found.** All four are either trained-direction + behavior-gated, training-free + concept-separability-gated, or training-free + alignment/entropy-gated on vision-language tasks — none gate training-free activation steering by the model's own answer-confidence on text reasoning/QA. Treat as reasonably thorough for Step 1, not exhaustive; worth a final targeted re-check closer to write-up. (Step 1.)
- [x] Does the gradient formula need ||w_ĉ|| normalisation? **Resolved 23 Oct 2026 (D-009): no.** The missing ||w_ĉ||^2 term is intrinsic to the paper; implement as printed and flag the discrepancy in the write-up. (Step 3.)
- [ ] Are MedQA, PubMedQA, MedMCQA sizes and licences as assumed? (Step 5.)
- [ ] Will Llama-3.2-1B score above chance on MedQA? (Step 5.)
- [ ] Gate on layer-wise g or final-layer softmax? (Step 6.)
- [ ] Does the professor expect a RAG or fine-tuning comparison? (Ask when sharing the proposal.)
- [ ] Which Colab GPU is available in practice, and what is steered-run throughput? (Step 2.)
- [x] Are arXiv:2603.13366 ("LEAD") and arXiv:2511.10292 ("RUDDER") close enough to our proposal to count as novelty threats? **Resolved 24 Oct 2026 (D-013): no.** Both training-free and adaptively gated, but neither gates on the model's own answer-confidence (LEAD gates on token entropy for embedding-mixing + a fixed visual anchor; RUDDER gates on visual-grounding alignment, with reinforcing-when-trusted logic), and both are vision-language, not text reasoning/QA. Cite both explicitly as closest analogs.
- [x] Does the paper's validation set (used to sweep α) overlap with its evaluation set? **Confirmed 23 Oct 2026, verbatim from raw paper text (direct read, not AI-summarized):** "We evaluate on the CSQA validation set (1,221)... the StrategyQA dev set (229)... the GSM8K test set (1,319)... the AQuA validation split (254)... The steering strength α is optimized per dataset using a validation set." No separate held-out split is described anywhere — α is tuned on the same split the final numbers are reported on, for all four datasets. Likely a known convention weakness in this subfield (training-free methods, no weight-fitting risk argued), not evidence of deception. **Action:** tune α on our own disjoint validation split, never on the set we report final accuracy on. (Step 4.)

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Gains fall within noise | Larger evaluation sets, paired tests, multiple seeds under sampling. A null result is still reportable. |
| Small models near chance on medical sets | Check baselines in Step 5; switch to MedMCQA or a larger model. |
| Medical sets test knowledge, not reasoning | State this limit; interpret null results carefully. |
| Cannot reproduce paper numbers | Document the mismatch; check hook placement, normalisation, and prompt format first. |
| Paper's α is tuned on the same split its final numbers are reported on (confirmed, not just suspected — no disjoint validation set is described) | Use our own clean, disjoint validation/test split; re-tune α ourselves rather than trusting Section 4.3 targets blindly. Frame as a subfield convention weakness, not an accusation of misconduct. |
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

- Liu, L., Hou, Q., Jian, Y., Pourreza, R., Ghavamzadeh, M., Memisevic, R., Qin, Y., Cai, H. Alleviating Hallucination in Reasoning Tasks with Training-Free Uncertainty-Guided Steering. arXiv:2609.38962, 2026 (NeurIPS 2026; Qualcomm AI Research, UC Santa Barbara). **(Core paper; full text independently re-checked 23 Oct 2026 — see D-009.)**
- Tripathi, H., Neupane, S., Mitra, S., Mittal, S., Amiri Golilarz, N., Rahimi, S. Gated Activation Steering for Reducing Sycophancy and Hallucination in Medical Question Answering. arXiv:2608.23666, 2026. **(Read in full 21 Oct 2026; related work, not a baseline — see Section 4.4 and D-007.)**
- Fereidouni, M., Haider, M. U., Sajjad, H., Siddique, A. B. GAPS: Dimension-Level Gates for Conditional Activation Steering. arXiv:2609.01878, 2026. **(Found 21 Oct 2026; training-free and gated, but different gate signal and domain — see Section 4.4 and D-008.)**
- Su, S., Zou, X., Feng, W., Mahapatra, D., Peng, Y., Lin, M., Ge, Z. Thinking in Uncertainty: Mitigating Hallucinations in MLRMs with Latent Entropy-Aware Decoding (LEAD). arXiv:2603.13366, 2026. **(Read in full 24 Oct 2026; closest "training-free + own-uncertainty-gated" analog found, still related work not a novelty threat — see Section 4.4 and D-013.)**
- RUDDER: Residual-Update Directed DEcoding Regulation. arXiv:2511.10292, 2026 (Aalto University; Chinese Academy of Sciences). **(Read in full 24 Oct 2026; training-free adaptive gate, but gates on visual-grounding alignment not answer-confidence — see Section 4.4 and D-013.)**
- Chapra, S., Kumar, D., Mandal, M., Sinha, Y. The Complexity Ceiling Benchmark. CTB@ICML 2026. **(Used for Option B; possible later extension for Option A.)**
- Chang, T. A., Arnett, C., et al. Global PIQA. arXiv:2510.24081. **(Not central; possible later multilingual robustness check.)**
- Baselines named in the USteer paper, confirmed 23 Oct 2026 against their own original papers (not just USteer's characterization):
  - Chuang, Y.-S., Xie, Y., Luo, H., Kim, Y., Glass, J., He, P. DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models. arXiv:2309.03883, ICLR 2024. Contrastive decoding — contrasts final-layer logits against an early-exit layer's logits. No steering vector, no learned direction, nothing training-free vs. not to debate — it never touches hidden states, only the final logit distribution.
  - Li, K., Patel, O., Viégas, F., Pfister, H., Wattenberg, M. Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. arXiv:2306.03341, NeurIPS 2023. Learns one fixed "truthful" direction from a few hundred labeled contrastive examples, applied to a small set of attention heads, same vector added at every decoding step regardless of context.
  - Zhang, S., Yu, T., Feng, Y. TruthX: Alleviating Hallucinations by Editing Large Language Models in Truthful Space. arXiv:2402.17811, ACL 2024. Autoencoder splits representations into semantic/truthful latent spaces; contrastive learning finds one truthful editing direction; edits all hidden layers (not just attention), still a single static learned vector.
  - **Net read:** all three predate USteer (2023–2024) and are not novelty threats — they're exactly the "trained, static-direction" category USteer (and our gating proposal) are positioned against. Confirms, rather than just assumes, D-002's framing.
- **Candidates found 23 Oct 2026, read in full 24 Oct 2026 from raw source text — the two closest novelty-adjacent papers found so far:**
  - **arXiv:2603.13366, LEAD ("Latent Entropy-Aware Decoding")**, Su, Zou, Feng, Mahapatra, Peng, Lin, Ge (Monash, Northeastern, Khalifa, Cornell, Minnesota). Training-free, plug-and-play decoding strategy for multimodal large reasoning models (MLRMs). Genuinely gated by the model's own token-level entropy with a *dynamically tracked* threshold (not fixed) — under high entropy it swaps the discrete one-hot next-token embedding for a probability-weighted mixture of all token embeddings, and injects a visual-anchor vector. **Classification: related work, closest found for "training-free + gated by own uncertainty," but not a novelty threat** — the injected vector is a fixed, pre-computed average of special vision-token embeddings (not a gradient-derived direction), the intervention is embedding-mixing rather than activation steering, and the domain is vision-language hallucination, not text reasoning/QA.
  - **arXiv:2511.10292, RUDDER ("Residual-Update Directed DEcoding Regulation")**, incl. authors from Aalto University and Chinese Academy of Sciences. Training-free; injects a per-sample "CARD" vector (extracted from the model's own prefill residual-stream activations, not learned offline) into hidden states during decoding for vision-language models, modulated by an adaptive "Beta Gate." **Classification: related work, not a novelty threat** — the gate signal is s_t = cos(hidden state, CARD vector), i.e. *visual-grounding alignment*, not the model's own answer-confidence; the gating logic also runs opposite to our hypothesis (RUDDER reinforces steering *more* when already aligned/trusted, backs off when not, vs. our "steer more when uncertain"); domain is vision-language object hallucination, not text reasoning.
  - **Net effect on novelty (D-002):** holds after four related papers checked (2608.23666, GAPS, LEAD, RUDDER). No paper found yet that is training-free, gates specifically on the model's own answer-confidence (via logit lens or equivalent), and applies to text reasoning/QA. LEAD and RUDDER should both be cited explicitly as the closest architectural analogs (training-free + adaptive per-token gating), to pre-empt a reviewer raising either.

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
| 23 Oct 2026 | Independently re-read core USteer paper (arXiv:2609.38962) full text. Confirmed g(z)'s missing ||w_ĉ||^2 term is intrinsic to the paper (D-009). Confirmed Appendix C greedy-decoding significance gap and found new detail: Section 4.5 claims significance under stochastic decoding — scoped RQ1 to greedy decoding plus a McNemar test. Added open question and risk about possible validation/test-set overlap for α tuning. |
| 23 Oct 2026 | Re-verified the two findings above against the raw paper HTML directly (D-010): greedy-vs-stochastic gap is openly disclosed by the authors, not an undisclosed flaw; validation/test overlap is now confirmed verbatim rather than suspected. Reframed both to avoid overclaiming in the eventual report. |
| 23 Oct 2026 | Confirmed DoLa/ITI/TruthX baseline descriptions against their original papers (D-011); not novelty threats, all predate USteer. Found two unread candidate papers that may be closer novelty threats (D-012); Step 1's "fuller search" stays open until they're read. Updated Sections 9, 11, 14. |
| 24 Oct 2026 | Read arXiv:2603.13366 (LEAD) and arXiv:2511.10292 (RUDDER) from raw source text; both classified as related work, not novelty threats (D-013). Novelty claim D-002 holds after four related papers checked. Step 1 marked done; moving to Step 2 (environment and baselines). Updated Sections 4.4, 9, 11, 14 and header status. |
| 24 Oct 2026 | Added a candidate (not yet run) protocol for choosing the gate threshold τ and the gate signal in Step 6, to Section 4.4 — percentile-based τ on a disjoint validation split, AUROC comparison for signal choice. General explanations (McNemar's test, forward-hook mechanics) discussed but intentionally not logged — not project-specific decisions or findings. |

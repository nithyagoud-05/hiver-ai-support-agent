# Engineering & Evaluation Report: Spotify AI Support Agent
**Candidate:** Nithya Goud  
**Role:** SDE Intern — AI / Machine Learning  
**Target Brand:** Spotify (@SpotifyCares)  
**Dataset:** Customer Support on Twitter (Kaggle: thoughtvector/customer-support-on-twitter)

---

## 1. Problem Framing & Scope

### 1.1 What "Good" Means for Spotify Support
Customer support in digital music streaming falls into two operational classes:
1. **Self-serviceable client-side issues:** Cache corruption, local storage limits, offline sync toggles, playlist recovery, and hardware connection setups.
2. **Account-bound and security-sensitive operations:** Double billing, unauthorized access, subscription tiers (Student/Family), and geographical licensing blackouts.

A production-grade AI support agent must maximize **trusted auto-handling** for self-serviceable software diagnostics while enforcing **deterministic escalation guardrails** for private, financial, or low-confidence interactions.

### 1.2 What We Chose NOT to Build
* **Autonomous CRM/Financial Execution:** The agent does not attempt to execute refunds, cancel subscriptions, or perform password updates over public Twitter threads.
* **Unconstrained Generative Hallucination:** Free-form generation was replaced by retrieval-grounded SOP resolution templates to eliminate the risk of committing to unavailable features or incorrect licensing timelines.

---

## 2. Intent Taxonomy & Golden Evaluation Set

### 2.1 8-Class Intent Taxonomy
Derived empirically from 25,005 reconstructed Spotify customer-agent interactions:
* `PLAYBACK_CRASH`: App freezing, shuffle bugs, playback buffer, audio stuttering.
* `OFFLINE_SYNC`: Greyed-out offline downloads, storage allocation, SD card sync.
* `SUBSCRIPTION_BILLING`: Overcharges, Family Plan invites, student discount renewals.
* `ACCOUNT_LOGIN`: Forgotten credentials, compromised accounts, Facebook auth migration.
* `CATALOG_CONTENT`: Regional music licensing, missing tracks/lyrics, metadata errors.
* `PLAYLIST_LIBRARY`: Recovering deleted playlists, missing Liked Songs, local files.
* `CONNECT_HARDWARE`: Bluetooth dropouts, CarPlay/Android Auto, Spotify Connect casting.
* `UNCLEAR_FEEDBACK`: Context-deficient rants, vague feedback, UI feature requests.

### 2.2 Golden Set Construction (200 Examples)
* **Sampling Strategy:** Stratified sampling across the 25,005 cleaned Spotify dialogues (~25 examples per intent) to prevent class imbalance.
* **Ground Truth Dimensions:** Hand-verified labels for `verified_intent`, `expected_escalation`, and `escalation_reason`.
* **Zero Leakage:** The 200 Golden Set items were isolated prior to training baseline models and populating the retrieval corpus.

---

## 3. Results vs. Baselines

| System / Model | Intent Accuracy | Intent Macro-F1 | Escalation Precision | Escalation Recall | Escalation F1 |
|---|---|---|---|---|---|
| **Trivial Baseline** (Majority Class: `UNCLEAR_FEEDBACK`) | 13.00% | 0.0288 | N/A | N/A | N/A |
| **Simple Baseline** (TF-IDF + Logistic Regression) | 75.50% | 0.7504 | N/A | N/A | N/A |
| **Production AI Agent** (Hybrid Classifier + RAG + Guardrails) | **75.50%** | **0.7504** | **85.19%** | **90.79%** | **0.8790** |

### Escalation Confusion Matrix
* **True Auto-Handle (TN):** 112
* **False Escalations (FP):** 12 (Safe over-escalation)
* **Missed Escalations (FN):** 7 (Critical risk cases)
* **True Escalations (TP):** 69

### LLM-as-a-Judge & Human Alignment (30-sample audit)
* **LLM Judge Mean Quality Score:** 4.64 / 5.0
* **LLM Judge Pass Rate:** 94.50%
* **Human <-> Judge Agreement:** **93.33%**

---

## 4. Top 5 Real Failure Modes & Root Cause Hypotheses

| Case ID | Customer Input | True Intent | Predicted Intent | Decision | Root Cause Hypothesis |
|---|---|---|---|---|---|
| **#5** | "This is happening on desktop. And on mobile it seems to be picking the same few artists again and again..." | `PLAYBACK_CRASH` | `CATALOG_CONTENT` | AUTO_HANDLE | Entity-keyword clash: the repeated mention of "artists" skewed n-gram weights toward catalog intent instead of the shuffle algorithm bug. |
| **#6** | "I have found the bug, What happens is when the playlist are downloading avast scans the files... then its deleted the files stuck in ram." | `PLAYBACK_CRASH` | `PLAYLIST_LIBRARY` | AUTO_HANDLE | Complex technical co-occurrence: customer uses "playlist" and "downloading" in the context of an antivirus memory deadlock. |
| **#28** | "Galaxy note 5, Android, and idk the spotify version. It's the latest app, as I tried uninstalling and redownloading" | `OFFLINE_SYNC` | `UNCLEAR_FEEDBACK` | ESCALATE | Context fragmentation: message was turn 2 of a thread containing purely device metadata without repeating the core symptom. |
| **#29** | "hey, hoping you can help - I have several thousand songs offline but for some reason random artist albums become unsynced" | `OFFLINE_SYNC` | `CATALOG_CONTENT` | AUTO_HANDLE | Semantic overlap between "artist albums" and "unsynced". Model prioritized catalog terms over sync state. |
| **#35** | "My music keeps undownloading, what is happening??? Please help" | `OFFLINE_SYNC` | `UNCLEAR_FEEDBACK` | ESCALATE | Out-of-vocabulary morphological variant: "undownloading" was not weighted in the n-gram dictionary, triggering safety fallback escalation. |

---

## 5. What is Misleading About My Headline Number?

1. **Stratified Sampling Inflates Rare Intent Performance:** The 200 Golden Set contains ~12.5% representation per intent. In live production Twitter streams, `UNCLEAR_FEEDBACK` and `PLAYBACK_CRASH` account for over 60% of volume. Live accuracy will depend heavily on boundary precision for those two categories.
2. **Single-Turn Thread Truncation:** Multi-turn customer follow-ups (like Case #28) lack explicit intent tokens when evaluated out of context. Single-turn accuracy masks degradation on multi-turn conversations.
3. **Escalation Recall Overstates Safety:** While 90.79% recall on `ESCALATE` is strong, the 7 False Negatives (3.5% of total volume) represent instances where the bot attempted to resolve an issue that required account verification or private tooling.
4. **Historical Twitter Data Quality:** Up to 35% of historical agent replies were generic DM redirections. Retrieval grounding against raw historical tweets without SOP filtering introduces boilerplate noise.

---

## 6. What I Would Do Next With One More Week

* **Dense Semantic Embeddings + Re-Ranking:** Replace TF-IDF retrieval with modern semantic embeddings plus a Cross-Encoder re-ranker to resolve morphological gaps like "undownloading".
* **Conversation State Tracking (Multi-Turn Context):** Concatenate previous turn context to prevent misclassifying diagnostic replies (e.g., "iPhone 12, iOS 16") as unclear feedback.
* **Dynamic Few-Shot In-Context Prompting:** Feed top-2 retrieved historical pairs directly into an LLM context window with strict schema outputs.
* **Automated SOP Knowledge Base Ingestion:** Index official Spotify Community Knowledge Base articles rather than relying on raw Twitter agent replies.

---

## 7. Decision Log (12 Non-Obvious Decisions)

1. **Selected SpotifyCares over AmazonHelp & AppleSupport:** Spotify had the lowest rate of generic DM deflections (35% vs 52% for Apple), yielding far richer public troubleshooting conversations.
2. **Excluded Private DM Chains from Auto-Resolution:** Avoided fabricating resolution steps for billing issues that can only be solved within Spotify's internal CRM.
3. **8-Class Bounded Taxonomy:** Merged sub-issues (e.g., family plan invites, student discounts) into `SUBSCRIPTION_BILLING` to avoid sparse training distributions.
4. **Hard Escalation for Sensitive Categories:** Hardcoded `SUBSCRIPTION_BILLING` and `ACCOUNT_LOGIN` to escalate regardless of classifier confidence to prevent security risks.
5. **0.40 Confidence Escalation Threshold:** Empirically selected to route out-of-vocabulary or low-confidence queries to human agents.
6. **Strict Train/Test Isolation:** Isolated the 200 Golden Set samples before generating the training corpus to guarantee zero evaluation leakage.
7. **Silver-Label Rule Bootstrapping for Corpus:** Applied heuristic intent tagging on 24,805 dialogue pairs to power intent-filtered vector retrieval.
8. **Top-K Intent Constrained Retrieval:** Constrained similarity search to historical cases within the predicted intent class to prevent cross-domain resolution hallucination.
9. **Single-Turn Standardization:** Stripped user Twitter handles and link trackers to evaluate purely on semantic content.
10. **LLM Judge Evaluation on 3 Axes:** Evaluated on Correctness, Groundedness, and Brand Tone rather than arbitrary single-score ratings.
11. **Dual Failure-Mode Categorization:** Split failures into *Safe Over-Escalations* (FP) and *Dangerous Under-Escalations* (FN) to reflect real business risk.
12. **Self-Contained Local Execution Pipeline:** Avoided external paid API requirements in the primary benchmark script so evaluators can reproduce results in under 15 seconds.

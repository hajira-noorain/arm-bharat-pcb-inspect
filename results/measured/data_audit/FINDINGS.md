# Llama training data: audit (2026-11-16)

Produced by `python -m llm.data.audit --out results/measured/data_audit` (numbers in `audit.json`, a hand-reading
sample in `sample.md`). Run after the first LoRA training run, before the next one.

## What the data is

- **Cause data** is synthetic. A generator draws a cause from the rule table (`kb/evidence.yaml`), then clues from
  that cause's likelihoods. Each example's label is the exact probability over causes that the table gives.
  6,000 training examples, plus 600 each for dev, test and the second generator.
- **Sentence data** is synthetic: rule-written fill-in sentences for the five shift-question types. There are
  3,000 of them, and every one passes the station's checker.
- **Real data**: 8 published real cases, used only for testing. No factory logs are used.

## What is fine

- Every defect type has 337–440 examples, and every one of the 42 causes has at least 60.
- None of the test clue combinations appears in training, and no held-out question appears in the sentence training data.
- Every cause is the most likely one for at least one clue combination, so the model can name any of them.

## Problems found

1. **The labels are our own estimates.**
   - 52% of training examples come from a cause whose rule is `inferred` (3,139 of 6,000).
   - Even for the 19 `sourced` rules, only the direction comes from the source; the numbers are ours.
   - Nobody with shop-floor experience has reviewed them. They go to the team through the Cause Rule Review page.
2. **The clues decide the cause only weakly.**
   - Picking the most likely cause is right 66.6% of the time on the training data. That is the ceiling.
   - The median top probability is 0.64, and 19% of examples have no cause above 0.5.
   - For changed part, unknown anomaly, foreign object and scratch, the ceiling is below 60%.
   - **Fix:** more clues, not more data. Tuned Llama already matches this ceiling: 62.2% on test, against 61.8% for the plain Bayes table.
3. **The cause lists have gaps.**
   - Mouse bite and spur list no phototool or artwork cause. So a defect that repeats at the same spot is credited to development, etch or adhesion, which happen across the whole panel, not at one fixed point.
   - Examples: sample #1, #15 and #30.
   - Same-spot repetition is our best-sourced clue (Dietz 2011), so this matters.
4. ~~Shape and edge clues decide part-defect causes.~~ **Wrong on checking.** Shape already has the same
   likelihood for every part-defect cause (except a solder ball, which really is round). In sample #6 the answer moved
   because the defect was not at the same spot, which is a legitimate clue.
5. **The dev split overlaps training.**
   - 371 dev clue combinations also appear in training, so early stopping measures fit on combinations the model has already seen.
   - **Fix:** draw dev from its own held-out slice (a second hash bucket), separate from test.
6. **The sentence targets are repetitive and dodge some questions.**
   - 3,000 examples have only 86 distinct targets, and the top five targets make up 48% of the data.
   - Comparison and rate questions are not answered:
     - "Were there more major or minor defects?" gets a list of counts.
     - "What is the pass rate?" gets "Out of 8 boards, 8 passed".
   - The facts have no comparison or rate fields, and the checker does not test whether the question was answered.
   - **Fix:** add `pass_rate_pct` and `more_of` fields, require them for those question words, and write more target phrasings.
   - Also collect real operator phrasings from the team.
7. **There is too little real data to trust.**
   - With 8 real cases, 5 of 8 right and 7 of 8 right are too close to tell apart.
   - **Target:** about 25–30 cases from application notes and failure-analysis papers.

## Changes made (2026-11-16)

- **Rules (finding 3).** Added a phototool cause to mouse bite (MBITE-C3) and spur (SPUR-C3), sourced to Dietz 2011.
  Lowered same-spot for the development and adhesion causes, which act across the whole panel.
- **Rules, from a published case.** Added MBITE-C4, bubbles in pattern plating (Carano 2018, confirmed by switching
  air spargers to eductors).
  - None of the station's clues tell it apart from over-etching, so it is never the top answer (the audit lists it
    under `causes_never_most_probable`).
  - Its confirm check (rounded bites, plating agitation) is what separates them.
- **Dev split (finding 5).** Clue combinations are now hashed into separate train, dev and test buckets. Dev overlap
  with train is 0.
- **Sentences (finding 6).**
  - New station-computed fields:
    - `pass_rate_pct`;
    - `comparison` ("more minor than major", "as many opens as shorts").
  - Rate and comparison questions now require those fields, and the standard sentence includes them.
  - Targets have more phrasings: 217 distinct (was 86), and the top five targets are 30% of the data (was 48%).
- **Real cases (finding 7).** A search for more published cases found only 2 more, both shifted LEDs (SMTA 2021).
  Their fit to "part turned or standing up" is loose, so they were not added. The real set stays at 8.
  - The search also found that real-2's source calls its cause suspected, not confirmed.
  - Growing this set needs paywalled or print sources (SMTA, APEX, IPC proceedings).
- **Baselines after the changes** (1000 per set, `cause_bayes.json` / `cause_prior.json`):

  | Scorer | test top-1 / top-2 | alt top-1 / top-2 | real top-1 / top-2 |
  |---|---|---|---|
  | Bayes | 60.0% / 87.6% | 55.8% / 85.3% | 5/8 / 7/8 |
  | Prior | 40.6% / 74.5% | – | – |

  The ceiling dropped slightly because mouse bite and spur now have more causes to choose between.

## Order of work (original)

1. Collect the rule review.
2. Fix the cause lists (3) and the part-defect clues (4) in `kb/evidence.yaml`.
3. Split dev properly (5).
4. Add the sentence rate and comparison fields and more phrasings (6).
5. Grow the real cases (7).
6. Retrain, sizing the data from a learning curve.

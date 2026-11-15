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
4. **Clues that probably mean nothing are scored for part defects.**
   - Shape and board edge carry likelihoods for missing, changed and extra parts.
   - For a component, the shape of the box is the shape of the part, not a clue to the cause.
   - Example: sample #6, where "round" moves the answer toward "part turned or standing up".
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

## Order of work

1. Collect the rule review.
2. Fix the cause lists (3) and the part-defect clues (4) in `kb/evidence.yaml`.
3. Split dev properly (5).
4. Add the sentence rate and comparison fields and more phrasings (6).
5. Grow the real cases (7).
6. Retrain, sizing the data from a learning curve.

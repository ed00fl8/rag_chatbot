# Retrieval evaluation

## Official documentation corpus

Download and index the fixed public documentation set:

```bash
uv run rag-assistant corpus
```

This downloads four pages from their official project documentation sources, saves them under `data/corpus/official/`, records each content SHA-256 and HTTP metadata in `manifest.json`, then indexes the files.

The upstream documentation is mutable. The manifest identifies the exact bytes downloaded on a run and records observed hash history. Historical hashes are metadata only; older page contents are not archived. A later download can produce a different snapshot.

## Seed benchmark

`benchmark.official.seed.jsonl` contains 60 initial retrieval queries: 15 per documentation source, including English and Spanish formulations. The `relevant_sources` labels are initial source-level annotations and must be audited against the exact downloaded snapshot before using the scores as research or CV claims. They are not a human-reviewed answer-faithfulness benchmark.

Run only after downloading and indexing the corpus:

```bash
uv run rag-assistant evaluate eval/benchmark.official.seed.jsonl --k 5
```

The evaluator measures source-level Recall@k, MRR and warm retrieval latency. It does not measure answer faithfulness, citation entailment, abstention accuracy, prompt-injection robustness, or end-to-end latency.

Do not report the seed benchmark's scores as definitive model performance until each query's relevant source has been checked against the downloaded source snapshot and the set has been held out from development/tuning.

## Evaluation boundaries

- Retrieval metrics require a representative, independently labeled set.
- Citation precision requires checking whether cited passages support each claim.
- Faithfulness requires claim-level evidence assessment, ideally with human review.
- Abstention evaluation needs a separate set of unanswerable questions and expected-abstention labels.
- Prompt-injection tests are scenario-specific and do not establish general security.

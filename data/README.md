# Processed Data

## Source

The source corpus is the English portion of OpenAssistant Conversations (`OpenAssistant/oasst1`) from Hugging Face. Please consult the dataset card for its current license and terms.

Raw data are not redistributed here. The processed files can be regenerated with:

```bash
python scripts/00_prepare_data.py
```

## Files

- `processed/oasst_typed_items.csv.gz`: reviewed, ranked English assistant responses with estimated contributor type, helpfulness label, and reliability score.
- `processed/oasst_preference_pairs.csv.gz`: preference pairs constructed from ranked sibling responses.
- `processed/oasst_test_pairs.csv.gz`: deterministic 800-pair held-out test split used by every policy.
- `processed/oasst_contributor_types.csv.gz`: contributor-level reliability and type estimates.
- `processed/type_summary.json`: type counts and bootstrap confidence intervals.

The split is deterministic and uses `seed=42`, with 80% training trees, 10% validation trees, and 10% test trees.

## Checksums

See `CHECKSUMS.sha256`.

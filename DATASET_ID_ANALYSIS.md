# Dataset Unique ID Analysis

## Summary

I investigated the datasets to check for unique identifiers for each problem.

| Dataset | Model | Has Unique ID? | ID Field Name | Example Value |
|---------|-------|----------------|---------------|---------------|
| **MATH500** | 70B | ✅ Yes | `unique_id` | `test/intermediate_algebra/776.json` |
| **MATH500** | 8B | ✅ Yes | `unique_id` | `test/intermediate_algebra/776.json` (Mapped) |
| **GPQA** | Both | ✅ Yes | `unique_id` | `gpqa/0` (Generated) |
| **MMLU** | Both | ✅ Yes | `unique_id` | `mmlu/0` (Generated) |

## Details

### MATH500
- **Field**: `unique_id`
- **Format**: Path-like string (e.g., `test/intermediate_algebra/776.json`)
- **Present in**: Both `hazyresearch` source and `wfang11` uploaded datasets.

### GPQA
- **Missing**: No `id`, `unique_id`, `problem_id`, or `index` field found.
- **Implication**: Problems are identified only by their content (question text) or their index in the dataset.

### MMLU
- **Missing**: No `id`, `unique_id`, `problem_id`, or `index` field found.
- **Implication**: Problems are identified only by their content (question text) or their index in the dataset.

## Recommendation

If you need consistent IDs across all datasets (e.g., for tracking specific problems), we could:
1.  **Add an index-based ID**: Create a new `unique_id` column for GPQA and MMLU based on the row index (e.g., `gpqa_0`, `gpqa_1`).
2.  **Use a hash**: Generate a hash of the question text to create a stable ID.

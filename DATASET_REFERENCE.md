# Dataset Upload Reference

## 📊 All Datasets Overview

**k value**: 100 (100 samples per problem)

**Unique IDs**:
- **MATH500**: ✅ Has `unique_id` (70B native, 8B mapped)
- **GPQA**: ✅ Has `unique_id` (Generated `gpqa/N`)
- **MMLU**: ✅ Has `unique_id` (Generated `mmlu/N`)

| Dataset | Model | Dev (1%) | Val (5%) | Test (94%) |
|---------|-------|----------|----------|------------|
| **MATH500** | Llama-70B | `wfang11/math500-llama70b-1-5-94-dev` | `wfang11/math500-llama70b-1-5-94-val` | `wfang11/math500-llama70b-1-5-94-test` |
| **MATH500** | Llama-8B | `wfang11/math500-llama8b-1-5-94-dev` | `wfang11/math500-llama8b-1-5-94-val` | `wfang11/math500-llama8b-1-5-94-test` |
| **GPQA** | Llama-70B | `wfang11/gpqa-llama70b-1-5-94-dev` | `wfang11/gpqa-llama70b-1-5-94-val` | `wfang11/gpqa-llama70b-1-5-94-test` |
| **GPQA** | Llama-8B | `wfang11/gpqa-llama8b-1-5-94-dev` | `wfang11/gpqa-llama8b-1-5-94-val` | `wfang11/gpqa-llama8b-1-5-94-test` |
| **MMLU** | Llama-70B | `wfang11/mmlu-llama70b-1-5-94-dev` | `wfang11/mmlu-llama70b-1-5-94-val` | `wfang11/mmlu-llama70b-1-5-94-test` |
| **MMLU** | Llama-8B | `wfang11/mmlu-llama8b-1-5-94-dev` | `wfang11/mmlu-llama8b-1-5-94-val` | `wfang11/mmlu-llama8b-1-5-94-test` |

**Total: 18 datasets** (6 combinations × 3 splits each)

## 🎯 Quick Commands

### Upload All Datasets
```bash
./scripts/upload_all_datasets.sh
```

### Load in Python
```python
from datasets import load_dataset

# MATH500 + Llama-70B
math500_70b_dev = load_dataset('wfang11/math500-llama70b-1-5-94-dev')['data']
math500_70b_val = load_dataset('wfang11/math500-llama70b-1-5-94-val')['data']
math500_70b_test = load_dataset('wfang11/math500-llama70b-1-5-94-test')['data']

# GPQA + Llama-8B
gpqa_8b_dev = load_dataset('wfang11/gpqa-llama8b-1-5-94-dev')['data']
gpqa_8b_val = load_dataset('wfang11/gpqa-llama8b-1-5-94-val')['data']
gpqa_8b_test = load_dataset('wfang11/gpqa-llama8b-1-5-94-test')['data']
```

## 🏷️ Naming Convention

```
wfang11/{dataset}-{model}-{split}-{dev|val|test}
```

- **{dataset}**: `math500`, `gpqa`, `mmlu`
- **{model}**: `llama70b` (70B), `llama8b` (8B)
- **{split}**: `1-5-94` (1% dev, 5% val, 94% test)
- **{dev|val|test}**: Split type

## 📈 Split Sizes

All datasets use **1-5-94 split**:
- Dev: 1% of original data
- Val: 5% of original data  
- Test: 94% of original data

## 🔗 View Your Datasets

After upload, view all your datasets at:
**https://huggingface.co/wfang11**


from datasets import load_dataset

def check_id(dataset_name, split="dev"):
    print(f"\nChecking {dataset_name} ({split})...")
    try:
        ds = load_dataset(dataset_name, split="data") # Uploaded datasets usually have 'data' split from parquet
    except:
        ds = load_dataset(dataset_name, split="train")
        
    if "unique_id" in ds.column_names:
        print(f"✓ Found 'unique_id' column")
        print(f"  Example: {ds[0]['unique_id']}")
        return ds
    else:
        print(f"❌ Missing 'unique_id' column")
        return None

print("Verifying IDs in uploaded datasets...")

# 1. Check MATH500 8B (should have mapped IDs)
math8b = check_id("wfang11/math500-llama8b-1-5-94-dev")
math70b = check_id("wfang11/math500-llama70b-1-5-94-dev")

if math8b and math70b:
    # Check if IDs look like paths (from 70B)
    if "/" in str(math8b[0]["unique_id"]):
        print("✓ MATH500 8B IDs look like paths (mapped correctly)")
    else:
        print("❌ MATH500 8B IDs do not look like paths")

# 2. Check GPQA (should have index IDs)
gpqa = check_id("wfang11/gpqa-llama8b-1-5-94-dev")
if gpqa:
    if str(gpqa[0]["unique_id"]).startswith("gpqa/"):
        print("✓ GPQA IDs have correct prefix")
    else:
        print(f"❌ GPQA IDs have wrong format: {gpqa[0]['unique_id']}")

# 3. Check MMLU (should have index IDs)
mmlu = check_id("wfang11/mmlu-llama8b-1-5-94-dev")
if mmlu:
    if str(mmlu[0]["unique_id"]).startswith("mmlu/"):
        print("✓ MMLU IDs have correct prefix")
    else:
        print(f"❌ MMLU IDs have wrong format: {mmlu[0]['unique_id']}")

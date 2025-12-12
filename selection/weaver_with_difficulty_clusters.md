## Vanilla Weaver (clusters already prepared)
- Adjust `selection/configs/math500_llm_judge_eval.yaml` for your run (e.g., `n_clusters`, `embedding_model` path).
- Run in cluster mode:  
  `python selection/run.py --config-path selection/configs --config-name math500_llm_judge_eval`
- Cluster IDs should be 0-indexed.

## Augmented Weaver (create clustered datasets end-to-end)
1) Build clustered splits and JSONs
```
python scripts/prepare_clustered_dataset.py \
  --main-dataset-base "wfang11/gpqa-llama8b-10-10-80" \
  --cluster-dataset "pnsahoo/GPQA_LLM_judge_fewshot_Qwen_Qwen2_5-7B-Instruct-Turbo" \
  --cluster-column "llm_difficulty_fewshot_Qwen_Qwen2_5-7B-Instruct-Turbo" \
  --output-base "pnsahoo/GPQA-llama8b-qwen-judge" \
  --push-to-hub
```
   - Updates: change dataset names/columns/output as needed.

2) Select per-cluster verifiers  
   `python verifier_selection/llm_judge_search.py`  
   - Update: `hub_name`, WANDB logging, CSV name/path.

3) Train Weavers with selected verifiers  
```
bash llm_judge_hyperparam.sh --config-name subset
```
   - Update: CSV from step 2, dataset path, log names/dev split, verifier column.

4) Final evaluation on test split  
```
bash final_eval_llm_judge.sh \
  --results-dir results \
  --dataset-path pnsahoo/GPQA-llama8b-qwen-judge-amended-test \
  --config-name subset \
  --log-name GPQA-llama8b-qwen-judge-amended \
  --log-dev test
```
   - Update: dataset path/log names/prefix if running on a different dataset.

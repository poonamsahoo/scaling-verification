# Monitoring Your Grid Search

Your full grid search is running! Here's how to monitor and manage it.

## 📊 Quick Status Check

```bash
# See which datasets completed
grep 'COMPLETED:' results/execution_log.txt

# Count completed datasets
grep -c 'COMPLETED:' results/execution_log.txt

# See current progress (last 50 lines)
tail -50 grid_search_full.log

# See errors (if any)
grep -i 'error\|failed' grid_search_full.log | tail -20
```

## 📈 Real-Time Monitoring

```bash
# Watch log in real-time (Ctrl+C to stop)
tail -f grid_search_full.log

# Watch only important events
tail -f grid_search_full.log | grep -E 'STAGE|COMPLETED|FAILED|accuracy'
```

## 📁 Check Results

```bash
# List all dataset directories
ls -la results/

# Check a specific dataset's progress
ls -la results/math500-llama70b/

# Quick peek at results
head results/math500-llama70b/weaver_all_configs_summary.csv
```

## 🔍 Detailed Progress

```bash
# See execution log with timestamps
cat results/execution_log.txt

# Count total configurations completed
find results/ -name "weaver_summary*.csv" | wc -l

# Expected: 486 total (27 configs × 18 datasets)
```

## ⏸️ Pause/Stop/Resume

```bash
# Find the process
ps aux | grep run_all_datasets_grid_search

# Stop gracefully (saves progress)
pkill -f "python run_all_datasets_grid_search.py"

# Resume (automatically skips completed datasets)
python run_all_datasets_grid_search.py --config datasets_config.yaml

# Force restart from beginning
python run_all_datasets_grid_search.py --config datasets_config.yaml --no-resume
```

## 📊 Generate Interim Results

You can generate results even while the grid search is still running:

```bash
# Aggregate completed datasets
python aggregate_results.py --results_dir results

# Generate report (with whatever is done so far)
python generate_report.py --results_dir results
```

## Expected Timeline

| Time | Datasets | Configs | What to Expect |
|------|----------|---------|----------------|
| 0-1h | 1-2 | 27-54 | First datasets completing |
| 3-6h | 6-12 | 162-324 | About halfway done |
| 9-18h | 18 | 486 | All completed! |

*Actual time varies based on dataset size and system load*

## 🎯 Success Indicators

You'll know it's working well when you see:
- ✅ "STAGE 1: VERIFIER SELECTION" messages
- ✅ "STAGE 2: WEAVER EVALUATION" messages  
- ✅ "COMPLETED: [dataset-name]" messages
- ✅ New directories appearing in `results/`
- ✅ Growing CSV files in dataset directories

## ⚠️ Warning Signs

Watch out for:
- ❌ Many "FAILED" messages
- ❌ Same dataset repeating (should auto-skip)
- ❌ No new output for > 2 hours
- ❌ Disk space warnings

## 🆘 Troubleshooting

### Out of Memory
```bash
# Check memory usage
top
# If high, reduce parallel operations or use smaller datasets
```

### Disk Space
```bash
# Check disk space
df -h
# Each dataset uses ~500MB-2GB
```

### Process Died
```bash
# Check if still running
ps aux | grep run_all_datasets_grid_search

# If not running, check exit code in log
tail -100 grid_search_full.log

# Resume
python run_all_datasets_grid_search.py --config datasets_config.yaml
```

## 📧 Get Notified When Done

```bash
# Add to end of command (Mac)
python run_all_datasets_grid_search.py --config datasets_config.yaml && osascript -e 'display notification "Grid search complete!" with title "Weaver"'

# Or setup email notification
python run_all_datasets_grid_search.py --config datasets_config.yaml && echo "Grid search complete" | mail -s "Weaver Complete" your@email.com
```

## 🎉 When Complete

Once you see all 18 datasets completed:

```bash
# 1. Aggregate all results
python aggregate_results.py --results_dir results

# 2. Generate final report
python generate_report.py --results_dir results

# 3. View results
cat results/report.md
open results/report.md  # Mac

# 4. Check WandB dashboard
# https://wandb.ai/329a/verification
```

## Quick Reference

| Command | Purpose |
|---------|---------|
| `tail -f grid_search_full.log` | Watch live progress |
| `grep 'COMPLETED:' results/execution_log.txt` | See completed datasets |
| `ls results/` | List all results |
| `python aggregate_results.py` | Combine results |
| `python generate_report.py` | Create final report |

---

**Started:** Check `grid_search_full.log` for start time
**Process ID:** Check with `ps aux | grep run_all_datasets`
**Log File:** `grid_search_full.log`
**Results:** `results/` directory


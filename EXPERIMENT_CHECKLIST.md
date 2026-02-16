
# Experiment 1 Checklist: No Dropout

Follow this guide to ensure experiment integrity and reproducibility.

## 1. Prerequisites (Code Updated)
I have already updated `src/train.py` and `src/utils.py`. The script now automatically:
- Creates `results/exp_no_dropout/`
- Saves `model.pth`, `loss_curves.png`, `confusion_matrix.png`, and `metrics.json` inside it.

## 2. Execution
Run the following command on your **GPU machine**:

```bash
python src/train.py --epochs 50 --batch_size 64 --lr 0.001 --dropout 0.0 --optimizer adamw --experiment_name exp_no_dropout
```

**Critical Rules:**
- [ ] Ensure it runs for **50 epochs** (or stops early automatically).
- [ ] Do **NOT** interrupt it manually unless it hangs.
- [ ] Do **NOT** change hyperparameters during the run.

## 3. Verification
After training is complete, check that the following exist:
- [ ] `results/exp_no_dropout/model.pth`
- [ ] `results/exp_no_dropout/loss_curves.png`
- [ ] `results/exp_no_dropout/confusion_matrix.png`
- [ ] `results/exp_no_dropout/metrics.json`

## 4. Documentation
Open `metrics.json` to get the exact numbers, then update `RESULTS.md`:

```markdown
## Experiment 1: No Dropout

- **Optimizer:** AdamW
- **Epochs:** 50
- **Dropout:** 0.0
- **Best Validation Accuracy:** [Insert from metrics.json]
- **Final Test Accuracy:** [Insert from metrics.json]

### Observations
- [Analyze the loss curves: Does train loss keep dropping while val loss goes up?]
- [Analyze the confusion matrix: Which classes are confused?]
```

## 5. Final Commit
Once `RESULTS.md` is updated and artifacts are generated:

```bash
git add results/exp_no_dropout/
git add RESULTS.md
git commit -m "Add experiment results: No Dropout (AdamW, 50 epochs)"
git push origin main
```

**Do NOT:**
- Push partial results.
- Push if the run didn't finish.

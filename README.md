# Diabetes Prediction

Small learning project. The notebook `diabetes prediction (1).ipynb` trains an SVM (SVC).

Data: Pima Indians diabetes data (diabetes.csv), predicting diabetes from 8 clinical features. The data files are not included in this repo, so you need to download them yourself.

## Benchmark script

`benchmark.py` compares SVM (RBF), random forest and HistGradientBoosting with 5-fold cross-validation.
It does not use the real dataset. It runs on synthetic data generated with numpy/scikit-learn (768 samples, 8 features), so the numbers below only compare the models on that data and say nothing about the real dataset.

## How to run

```
pip install numpy scikit-learn jupyter
jupyter notebook
python benchmark.py
```

## Results (5-fold CV, mean)

| Model | accuracy | f1 | roc_auc |
|---|---|---|---|
| SVM (RBF Kernel) | 0.8711 | 0.8024 | 0.921 |
| HistGradientBoosting | 0.8516 | 0.7683 | 0.9086 |
| Random Forest (200 Trees) | 0.8346 | 0.7292 | 0.9038 |

Full numbers are in `benchmarks/benchmark_results.json`.

## Files

- `diabetes prediction (1).ipynb`: notebook
- `benchmark.py`: benchmark script
- `benchmarks/benchmark_results.json`: results
- `assets/benchmark_chart.svg`: chart of the results

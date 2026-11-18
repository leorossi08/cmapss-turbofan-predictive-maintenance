"""RUL models: supervised regression of Remaining Useful Life from sensor readings.

Three families, all compared on the same engine-level splits (see benchmark.py):
  A. Tabular (one row -> RUL): Ridge, SVM, RandomForest, ExtraTrees, XGBoost, LightGBM, CatBoost, MLP.
     Cheap, strong baselines. Built by registry.get_model_pool; gbdt.py holds the XGBoost NASA-loss variant.
  B. Sequence (last 15 cycles -> RUL): LSTM/GRU, 1-D CNN, Transformer (sequences.py, networks.py, torch_models.py).
  C. MiniRocket + RidgeCV (minirocket.py): fixed random convolution kernels turn a window into features.

Rules every model here must respect (tests in tests/test_modeling.py):
  [BASELINE]   always compare against "predict the mean"; a model that cannot beat it learned nothing.
  [GROUPED-CV] GroupKFold by engine (registry.grouped_oof, benchmark.RULBenchmark.cross_validate).
  [LEAKAGE]    scalers are fit on the training fold only; sequences never cross engines;
               early stopping uses a held-out set of ENGINES, not rows.
  [CAUSALITY]  sequence windows look backwards only; early cycles are left-padded with the first reading.
  [METRICS]    R2 on a capped, mostly-flat target can look great; also read RMSE, the NASA score and
               the critical-zone (RUL < 50) table.
"""

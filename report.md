# 1. Motivation & Introduction

| Item | Information |
|---|---|
| Student name | 이승빈 |
| Student ID | 12243785 |

본 프로젝트의 목표는 18개 기지국에서 측정된 RTT 값을 이용하여 실내 사용자의 2차원 위치를 추정하는 것이다. 실내 환경에서는 GPS 사용이 제한적이며, 무선 신호는 벽, 기둥, 반사체, 다중경로, NLOS 조건의 영향을 크게 받는다. 따라서 RTT 값을 단순한 실제 거리로 가정하고 삼각측량이나 최소제곱법만 적용하면, 측정 bias와 비선형 왜곡 때문에 안정적인 위치 추정이 어렵다.

본 프로젝트에서는 이 문제를 거리 역산 문제가 아니라 RTT fingerprint 기반 위치 회귀 문제로 정의하였다. 즉, 각 사용자에 대해 18개 anchor에서 관측된 RTT 패턴 전체를 하나의 fingerprint로 보고, 이 fingerprint와 실제 위치 사이의 비선형 관계를 학습하였다. 이 접근은 개별 RTT 값의 물리적 해석이 완벽하지 않아도, anchor 간 상대적 크기, 순위, 차이, 비율이 위치 정보를 포함한다는 점을 활용한다.

무선 신호 기반 indoor positioning에서는 측정값을 절대 거리로 직접 변환하기보다, 위치별 신호 패턴을 fingerprint로 해석하고 machine learning 모델을 적용하는 접근이 널리 사용된다 [7], [8]. 본 프로젝트에서도 RTT 측정값과 실제 거리가 완전히 일치하지 않는 점을 고려하여, RTT 값을 위치별 fingerprint feature로 구성하였다.

중간 프로젝트에서는 kNN fingerprinting, WLS, Robust NLS, reliability weighting, fixed ensemble 등을 사용하였다. 그러나 단순히 여러 모델을 고정 가중 평균하는 방식은 왜 특정 weight를 선택했는지, 어떤 상황에서 어떤 모델을 신뢰해야 하는지에 대한 설명이 약할 수 있다. 따라서 기말 프로젝트에서는 중간 프로젝트의 알고리즘을 그대로 반복하지 않고, OOF validation을 통해 실제 일반화 성능이 확인된 구성만 최종 모델에 남기는 방향으로 설계하였다.

# 2. Algorithm Explanation

최종 알고리즘은 RTT Fingerprint OOF Stacking Ensemble이다. 입력 데이터는 `DH_FR1.mat`에서 읽으며, 주요 변수는 `d_hat`, `p`, `BS_positions`이다. `d_hat`은 18개 기지국에 대한 RTT 관측값이고, `p`는 학습용 실제 위치 좌표이다. 사용자 수는 코드에 고정하지 않고 `d_hat.shape[1]`로 동적으로 결정하였다. 최종 출력은 `(2, num_user)` 형태의 numpy array이다.

RTT 값은 실제 거리와 직접 일치하지 않을 수 있기 때문에, 단순 raw RTT만 사용하는 대신 여러 종류의 fingerprint feature를 구성하였다. Raw RTT와 log RTT는 전체적인 측정 크기를 반영하고, 정렬 feature는 어떤 anchor가 상대적으로 가까운지 또는 먼지를 나타낸다. Anchor 간 차이와 비율 feature는 절대적인 RTT scale보다 상대적인 패턴을 강조하므로 anchor별 bias나 scale distortion에 더 강건할 수 있다. 또한 RTT 기반 weighted centroid feature를 추가하여, 순수 데이터 기반 fingerprint와 anchor geometry 정보를 함께 반영하였다.

| Feature group | Meaning | Purpose |
|---|---|---|
| Raw RTT | Original RTT measurements | Preserve direct signal magnitude information |
| Log RTT | Compressed RTT scale | Reduce the influence of extremely large RTT values |
| Sorted RTT | Relative ordering of anchors | Capture nearest/farthest anchor pattern |
| Difference features | Pairwise RTT differences | Emphasize relative fingerprint structure |
| Ratio features | Pairwise RTT ratios | Improve robustness to scale distortion |
| Weighted centroid | RTT-weighted anchor geometry | Combine fingerprint pattern with spatial anchor layout |

Base model은 주로 HistGradientBoostingRegressor와 ExtraTreesRegressor 계열을 사용하였다. HistGradientBoostingRegressor는 tabular feature에서 비선형 관계를 안정적으로 학습할 수 있고, ExtraTreesRegressor는 다양한 feature interaction을 포착하는 데 유리하다. 각 feature mode와 model configuration은 5-fold OOF validation으로 평가하였다. Tree ensemble 계열 모델은 비선형 feature interaction을 학습할 수 있다는 장점이 있으며, 본 프로젝트에서는 random forest, extremely randomized trees, gradient boosting 계열 모델의 개념을 참고하여 RTT feature 기반 위치 회귀 모델을 구성하였다 [3], [4], [5].

최종 결합 방식은 OOF stacking이다. 각 fold에서 학습 fold로 base model을 학습하고, 해당 모델이 보지 않은 validation fold에 대해 위치를 예측하였다. 이렇게 얻은 OOF 예측 좌표를 meta-level 입력으로 사용하여 RidgeCV stacking을 수행하였다. 이 과정은 train data에 다시 예측한 결과를 성능으로 착각하는 것을 방지하고, hidden test에 가까운 일반화 성능을 추정하기 위한 목적이다. 여러 base model의 예측 결과를 다시 meta-model의 입력으로 사용하는 stacking ensemble은 stacked generalization 개념에 기반한다 [2]. 본 프로젝트에서는 train prediction을 그대로 사용하는 대신, 각 base model의 OOF prediction을 meta-level 입력으로 사용하여 과적합 위험을 줄이고자 하였다.

최종 예측은 RidgeCV stacking 결과와 OOF 성능이 우수한 top base model average를 함께 사용하였다. 이는 한 모델의 fold별 편향에 과하게 의존하지 않기 위한 안정화 장치이다. 중요한 점은 최종 조합이 임의의 고정 평균이 아니라, OOF 성능 비교를 통해 선택되었다는 것이다.

| Component | Final design |
|---|---|
| Problem formulation | RTT fingerprint-based 2D coordinate regression |
| Input | `d_hat`, `BS_positions` |
| Output | `(2, num_user)` estimated position |
| Validation | 5-fold OOF validation |
| Main features | raw, log, sorted, difference, ratio, weighted centroid |
| Base learners | HistGradientBoostingRegressor, ExtraTreesRegressor |
| Meta model | RidgeCV stacking |
| Final stabilization | Blend with top OOF base-model average |
| Saved model | `model.pkl.xz` |

# 3. Agent AI Usage

본 프로젝트에서는 ChatGPT와 Codex를 보조 도구로 활용하였다. ChatGPT는 알고리즘 방향 설정, 중간 프로젝트 피드백 해석, OOF 결과 분석, README 제출 조건 검토, 보고서 구성 점검에 사용하였다. Codex는 로컬 프로젝트 폴더에서 코드 구조 확인, 제한된 후보 실험 스크립트 작성, 추가 모델 튜닝 검증, 제출 폴더 조건 확인에 사용하였다.

AI는 최종 답을 자동으로 결정하는 도구가 아니라, 후보를 정리하고 검증 절차를 빠르게 구성하기 위한 보조 도구로 사용하였다. 실제 최종 모델 채택 여부는 OOF 성능, 모델 크기, 실행 시간, README 조건 만족 여부를 기준으로 결정하였다. 예를 들어 kNN fingerprinting, calibrated geometry, Robust NLS, Kernel Ridge 계열은 모두 후보로 검토되었지만, OOF 성능이 안정 모델보다 낮았기 때문에 최종 모델에는 포함하지 않았다.

| Tool | Usage | Final decision criterion |
|---|---|---|
| ChatGPT | Design review, feedback interpretation, report drafting | OOF result and README compliance |
| Codex | Local code inspection, bounded experiments, folder validation | Stable model comparison |
| User | Execution, result verification, final selection | Runtime, shape check, model size, OOF metric |

# 4. Results & Discussion

최종 성능 평가는 제공된 700개 학습 데이터에 대해 5-fold OOF 방식으로 수행하였다. OOF 평가는 각 sample이 해당 fold의 학습 과정에 포함되지 않은 상태에서 예측된 결과이므로, 단순 train fit 결과보다 hidden test 성능을 추정하는 데 더 적합하다.

최종 모델의 OOF 성능은 다음과 같다.

| Metric | Final OOF error |
|---|---:|
| Mean | 5.985227 m |
| RMSE | 7.080726 m |
| Median | 5.439129 m |
| P70 | 7.209263 m |
| P90 | 10.073312 m |
| P95 | 12.373707 m |
| Max | 31.913228 m |

Train fit 결과도 함께 확인하였지만, 이는 학습 데이터에 다시 적합한 결과이므로 최종 성능 주장에는 사용하지 않았다. 실제 성능 판단은 OOF 결과를 기준으로 하였다.

| Evaluation | Mean | RMSE | Median | P90 | Max |
|---|---:|---:|---:|---:|---:|
| Final fit train | 0.483726 m | 0.589108 m | 0.427527 m | 0.832315 m | 4.866028 m |
| Final blended OOF | 5.985227 m | 7.080726 m | 5.439129 m | 10.073312 m | 31.913228 m |

Base model 비교 결과, diff_ratio와 diff feature를 사용한 HistGradientBoostingRegressor 계열이 가장 안정적인 성능을 보였다. 이는 RTT의 절대값보다 anchor 간 상대적 차이와 비율이 위치 fingerprint를 설명하는 데 중요한 역할을 했음을 의미한다.

| Base model | Feature mode | OOF mean | OOF median | OOF P90 | OOF max |
|---|---|---:|---:|---:|---:|
| diff_ratio_HistGBR_a | diff_ratio | 6.165127 m | 5.503250 m | 10.617742 m | 30.125553 m |
| diff_HistGBR_a | diff | 6.189905 m | 5.749680 m | 10.375420 m | 30.766324 m |
| diff_ratio_HistGBR_b | diff_ratio | 6.202683 m | 5.594041 m | 10.563350 m | 30.122145 m |
| raw_log_sort_HistGBR_a | raw_log_sort | 6.291861 m | 5.574903 m | 10.739838 m | 33.975362 m |
| diff_HistGBR_b | diff | 6.293351 m | 5.801087 m | 10.950526 m | 30.432359 m |
| rawlog_ET_leaf1 | raw_log_sort | 6.368976 m | 5.735148 m | 11.097907 m | 36.262797 m |

최종 ensemble은 best single base model보다 평균 오차를 낮추었다. 아래 표에서 볼 수 있듯이 Ridge stacking과 top base model average는 서로 다른 장단점을 보였고, final blended ensemble은 두 예측의 보완성을 활용하였다. 이는 서로 다른 feature mode와 model family가 일부 다른 sample에서 보완적인 예측을 제공했기 때문으로 해석된다.

| Model configuration | Mean | RMSE | Median | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Best single base model | 6.165127 m | - | 5.503250 m | 10.617742 m | - | 30.125553 m |
| Ridge stacking | 6.013934 m | 7.137368 m | 5.520352 m | 10.223078 m | 12.267822 m | 33.072111 m |
| Top6 average | 6.064031 m | 7.145607 m | 5.550491 m | 10.157828 m | 12.506698 m | 30.762770 m |
| Final blended ensemble | 5.985227 m | 7.080726 m | 5.439129 m | 10.073312 m | 12.373707 m | 31.913228 m |

추가 후보 알고리즘도 검토하였다. 중간 프로젝트에서 사용했던 kNN fingerprinting, geometry-based WLS, Robust NLS는 직관적으로 해석 가능하다는 장점이 있지만, 이번 데이터셋에서는 최종 fingerprint ensemble보다 낮은 OOF 성능을 보였다. kNN fingerprinting baseline은 nearest neighbor 기반 pattern matching 개념을 바탕으로 하며, RTT 패턴이 유사한 학습 샘플의 위치를 이용해 예측하는 방식으로 구성하였다 [6]. 따라서 설명 가능한 고전적 방법을 억지로 포함하지 않고, 실제 검증 성능이 높은 모델을 최종 선택하였다.

| Additional experiment | Best OOF mean | Decision |
|---|---:|---|
| KNN local fingerprint | 7.837295 m | Not selected |
| Calibrated geometry / Robust NLS | 8.609900 m | Not selected |
| Kernel Ridge / PCA | Above 7 m | Stopped due to low efficiency |
| Additional HGB / compact ExtraTrees tuning | 6.041059 m | Not selected |
| Stable final ensemble | 5.985227 m | Selected |

Baseline 비교의 fairness도 고려하였다. 단순 삼각측량이나 WLS는 RTT가 실제 거리와 잘 대응된다는 가정이 강한 반면, 본 데이터의 d_hat은 실제 환경 왜곡을 포함한 RTT 관측값이다. 따라서 geometry-only baseline과 ML fingerprint model을 완전히 같은 조건의 모델로 비교하는 것은 공정하지 않을 수 있다. 본 프로젝트에서는 WLS와 Robust NLS를 최종 모델의 직접 경쟁 baseline으로 과도하게 주장하지 않고, RTT fingerprint 데이터에서 실제 OOF 성능이 낮았던 후보로 해석하였다. 최종 성능 판단은 동일한 학습 데이터와 동일한 5-fold OOF 절차를 적용한 model family 간 비교를 중심으로 수행하였다.

본 프로젝트에서 직접 설계한 핵심은 단순히 scikit-learn 모델을 적용한 것이 아니라, RTT 측정값의 불완전성을 고려하여 feature, validation, model selection을 함께 설계한 점이다. 특히 RTT를 실제 거리로 강하게 가정하지 않고, anchor별 상대적 패턴을 학습 가능한 fingerprint로 변환한 것이 최종 구조의 출발점이었다.

| Design choice | Why it was used | Difference from a simple baseline |
|---|---|---|
| RTT fingerprint formulation | RTT measurements may include bias, NLOS distortion, and scale mismatch. | Did not directly trust RTT as physical distance for pure multilateration. |
| Relative RTT features | Anchor ranking, pairwise differences, and ratios can remain informative even when absolute RTT values are biased. | Used relative structure instead of only raw RTT values. |
| Geometry-aware centroid features | Anchor positions still contain useful spatial information. | Used geometry as a feature, not as a hard WLS/NLS constraint. |
| OOF-based model selection | Train-fit performance was too optimistic for the small public dataset. | Selected models using unseen-fold predictions rather than in-sample accuracy. |
| Conservative ensemble design | More complex candidates were only kept when OOF improvement justified them. | Rejected kNN, WLS/NLS, KRR/PCA, and extra tuning when they did not improve OOF results. |

최종 모델의 장점은 네 가지이다. 첫째, RTT를 직접 거리로 역산하지 않고 fingerprint regression으로 접근하여 측정 bias와 NLOS distortion에 더 유연하게 대응하였다. 둘째, OOF validation을 사용하여 train fit overfitting을 최종 성능으로 오해하지 않도록 하였다. 셋째, 모델 조합을 임의의 고정 평균이 아니라 OOF 기반 stacking으로 구성하였다. 넷째, 여러 후보 알고리즘 중 실제 성능 개선이 확인된 구성만 최종 모델에 포함하였다.

한계도 존재한다. 학습 데이터가 제한되어 있으므로 hidden 데이터의 분포가 학습 데이터와 다르면 성능 차이가 발생할 수 있다. 또한 tree ensemble 기반 모델은 단순 기하학 모델보다 파일 크기가 크다. 이를 완화하기 위해 최종 모델은 `model.pkl.xz`로 압축 저장하였다. 압축 후 모델 크기와 `main.py` 실행 시간은 아래 제출 확인 표에 정리하였다.

| Submission check | Result |
|---|---:|
| Model file | model.pkl.xz |
| Model file size | 88,506,628 bytes |
| main output shape | (2, 700) |
| Finite value check | True |
| Local runtime | About 10 seconds |
| Runtime limit | Within 10 minutes |

향후 개선 방향은 세 가지이다. 첫째, 더 다양한 위치와 환경의 RTT 데이터가 확보된다면 hidden test 분포 변화에 대한 강건성을 높일 수 있다. 둘째, 예측 좌표뿐 아니라 sample-wise uncertainty를 함께 추정하면 tail error가 큰 sample을 별도로 식별할 수 있다. 셋째, anchor별 NLOS 가능성이나 환경 구조 정보를 추가로 사용할 수 있다면, 현재의 fingerprint feature와 결합하여 P90, P95, max error를 줄이는 방향으로 개선할 수 있다.

# 5. References

본 프로젝트는 특정 논문 하나의 알고리즘을 그대로 재현한 것이 아니라, indoor positioning, fingerprinting, tree ensemble, gradient boosting, nearest neighbor, stacking ensemble, scikit-learn 구현 문서를 참고하여 수업 제공 RTT 데이터에 맞게 feature와 model selection 절차를 직접 구성하였다. 아래 표는 각 reference에서 참고한 내용, 본 프로젝트에서 직접 설계 또는 구현한 부분, 그리고 보고서와 코드에서 연결되는 위치를 구분한 것이다.

| Reference | 참고한 내용 | 본 프로젝트에서 직접 설계 또는 구현한 부분 | 연결되는 보고서 또는 코드 부분 |
|---|---|---|---|
| [1] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011. URL: https://www.jmlr.org/papers/v12/pedregosa11a.html | Python 기반 machine learning 구현 도구인 scikit-learn의 기본 참고문헌이다. | HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor, KNeighborsRegressor, RidgeCV, KFold를 사용하여 OOF validation과 ensemble regression을 구현하였다. | `main.py`의 model import, `make_model_specs`, `fit_package`, `train.py`의 학습 및 model 저장 절차 |
| [2] D. H. Wolpert, "Stacked Generalization," Neural Networks, vol. 5, no. 2, pp. 241-259, 1992. DOI: 10.1016/S0893-6080(05)80023-1 | 여러 base model의 예측 결과를 meta-level 입력으로 사용하는 stacking ensemble 개념을 참고하였다. | Base model의 train prediction이 아니라 OOF prediction을 meta-level 입력으로 사용하고, RidgeCV를 이용해 최종 stacking model을 구성하였다. | Algorithm Explanation의 OOF stacking 설명, `fit_package`의 OOF prediction 생성과 `meta_model` 학습 |
| [3] J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine," The Annals of Statistics, vol. 29, no. 5, pp. 1189-1232, 2001. DOI: 10.1214/aos/1013203451 | Gradient boosting 계열 회귀 모델의 기본 개념을 참고하였다. | RTT fingerprint feature와 2D 위치 좌표 사이의 비선형 관계를 학습하기 위해 HistGradientBoostingRegressor 계열 모델을 후보로 구성하고 OOF 성능으로 선택하였다. | Algorithm Explanation의 base model 설명, Results의 HistGBR base model 비교, `make_model_specs`의 HistGBR 후보 |
| [4] L. Breiman, "Random Forests," Machine Learning, vol. 45, pp. 5-32, 2001. DOI: 10.1023/A:1010933404324 | 여러 decision tree를 결합하여 예측 안정성을 높이는 ensemble tree 모델의 기본 개념을 참고하였다. | RandomForestRegressor를 후보 모델로 검토하고, tree ensemble 계열 모델 간 OOF 성능을 비교하였다. | `make_model_specs`의 RandomForestRegressor 후보와 Results의 후보 알고리즘 선택 기준 |
| [5] P. Geurts, D. Ernst, and L. Wehenkel, "Extremely Randomized Trees," Machine Learning, vol. 63, pp. 3-42, 2006. DOI: 10.1007/s10994-006-6226-1 | ExtraTreesRegressor의 기반이 되는 extremely randomized trees 개념을 참고하였다. | 다양한 RTT feature 조합의 비선형 interaction을 학습하기 위한 후보 모델로 ExtraTreesRegressor를 사용하였다. | Algorithm Explanation의 ExtraTrees 설명, Results의 `rawlog_ET_leaf1` 비교, `make_model_specs`의 ExtraTrees 후보 |
| [6] T. Cover and P. Hart, "Nearest Neighbor Pattern Classification," IEEE Transactions on Information Theory, vol. 13, no. 1, pp. 21-27, 1967. DOI: 10.1109/TIT.1967.1053964 | Nearest neighbor 기반 pattern matching 개념을 참고하였다. | RTT fingerprint가 유사한 학습 샘플의 위치를 이용해 예측하는 kNN fingerprinting baseline을 비교 대상으로 사용하였다. | Results의 additional experiment 표, `make_model_specs`의 KNeighborsRegressor 후보, 별도 kNN 실험 결과 |
| [7] V. Bellavista-Parent, J. Torres-Sospedra, and A. Perez-Navarro, "New trends in indoor positioning based on WiFi and machine learning: A systematic review," arXiv:2107.14356, 2021. URL: https://arxiv.org/abs/2107.14356 | 무선 신호 fingerprint와 machine learning을 이용한 indoor positioning 연구 흐름을 참고하였다. | RTT 측정값을 단순 거리값이 아니라 위치별 fingerprint로 해석하고, feature engineering과 OOF 기반 model selection으로 최종 위치 회귀 모델을 설계하였다. | Motivation의 RTT fingerprint 문제 정의, Algorithm Explanation의 feature engineering 설명 |
| [8] P. Bahl and V. N. Padmanabhan, "RADAR: An In-Building RF-Based User Location and Tracking System," IEEE INFOCOM, pp. 775-784, 2000. DOI: 10.1109/INFCOM.2000.832252 | 실내 무선 측정값을 위치 추정을 위한 fingerprint로 활용하는 초기 indoor localization 연구를 참고하였다. | RADAR의 RSS radio map을 그대로 구현하지 않고, 제공된 RTT `d_hat`을 anchor별 fingerprint로 해석하여 2D 위치 회귀 feature를 직접 구성하였다. | Motivation의 fingerprint 접근 설명, `build_features`의 raw, sorted, difference, ratio RTT feature 구성 |

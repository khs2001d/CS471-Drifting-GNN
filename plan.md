# Anti-Symmetric Drifting 복원 계획

## 핵심 목표

현재 student 학습 loss는 teacher sample 쪽 attraction만 사용한다. 이번 수정의 목표는 임의의 diversity regularizer를 추가하는 것이 아니라, Drifting 논문의 핵심 구조인 positive/negative particle field를 DiffSTG teacher-student 설정에 맞게 복원하는 것이다.

```text
teacher samples = positive particles
student samples = generated / negative particles
V = attraction_to_teacher - repulsion_from_student
loss = mse(student, stopgrad(student + eta * V))
```

## 구현 원칙

- 기존 `rbf_drift_to_teacher_loss`는 attraction-only baseline으로 유지한다.
- 새 `antisymmetric_drift_to_teacher_loss`를 추가해 Drifting core method를 구현한다.
- 1차 구현에서는 새 CLI를 `--loss_type attraction|antisymmetric` 하나만 추가한다.
- `eta`, `sigma`, `num_student_samples`는 기존 인자를 그대로 재사용한다.
- `temperature`, `lambda_rep`, `lambda_mmd`, `lambda_div`, checkpoint selection 변경은 1차 구현 범위에서 제외한다.

## 이유

현재 collapse 문제를 해결하는 가장 정체성 있는 방법은 별도 diversity penalty를 붙이는 것이 아니라, 원래 Drifting 알고리즘의 negative/generated particles를 loss field에 포함하는 것이다. 이렇게 해야 teacher distribution으로 끌어당기는 힘과 student/generated distribution에서 밀어내는 힘이 함께 작동하고, attraction-only baseline과의 비교도 명확해진다.

## 실험 순서

1. 기존 baseline 재현

```text
python train_drift_teacher_student.py --loss_type attraction
```

2. Drifting core method 평가

```text
python train_drift_teacher_student.py --loss_type antisymmetric
```

3. 두 결과를 아래 지표로 비교한다.

```text
MAE
RMSE
CRPS
sample_diversity
inference_time
```

## 후속 ablation

1차 결과를 본 뒤에만 다음을 추가 검토한다.

- repulsion 강도 조절
- multi-temperature kernel
- feature-space drift
- CRPS/diversity 기반 checkpoint selection
- MMD 또는 diversity floor 보조항

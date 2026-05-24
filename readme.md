# CS471 Drifting GNN

## Project Overview

This project studies how to distill a diffusion-based spatio-temporal graph forecasting model into a one-step inference model. We use DiffSTG as the baseline teacher model and train a Drift-to-teacher student model that can generate probabilistic forecasts with a single network evaluation.

The overall pipeline is:

1. Train a DiffSTG teacher model on the PEMS08 traffic dataset.
2. Use the trained teacher to generate forecast sample caches for the train and validation splits.
3. Train a one-step Drift-to-teacher student model from the cached teacher samples.
4. Fine-tune the student with an energy-distance auxiliary loss to improve distribution matching and sample diversity.
5. Evaluate teacher and student models on fresh test samples using accuracy, uncertainty, diversity, and inference speed metrics.

The teacher uses multi-step DDIM sampling. The student uses only one UGnet forward pass, so its number of function evaluations is `NFE=1`.

## Dataset

This project uses the PEMS08 traffic flow dataset.

Expected data layout:

```text
data/
  dataset/
    PEMS08/
      flow.npy
      adj.npy
```

File descriptions:

- `flow.npy`: Traffic flow values for all sensors over time. The code uses the first feature channel. The expected shape is approximately `(time, sensors, features)`.
- `adj.npy`: Graph adjacency matrix for the 170 PEMS08 sensor nodes.

Dataset configuration:

- Dataset: `PEMS08`
- Number of vertices: `170`
- Number of features: `1`
- History length `T_h`: `12`
- Prediction length `T_p`: `12`
- Points per hour: `12`
- Train/validation/test split: 60% / 20% / 20% over 17856 time steps
- Normalization: standardization using the training split mean and standard deviation

## Code Structure

```text
.
├── train.py
├── generate_teacher_cache.py
├── train_drift_teacher_student.py
├── eval_teacher_pems08.py
├── eval_drift_teacher_student.py
├── generate_fresh_pems08_samples.py
├── eval_fresh_distribution_comparison.py
├── algorithm/
│   ├── dataset.py
│   ├── diffstg/
│   │   ├── model.py
│   │   ├── ugnet.py
│   │   ├── graph_algo.py
│   │   └── checkpoint.py
│   └── drifting/
│       ├── student.py
│       └── losses.py
├── utils/
│   ├── common_utils.py
│   ├── eval.py
│   └── gpu_dispatch.py
└── scripts/
```

Main files:

- `train.py`: Entry point for DiffSTG teacher training.
- `generate_teacher_cache.py`: Generates train/validation teacher sample caches from a trained teacher checkpoint.
- `train_drift_teacher_student.py`: Trains the one-step student from cached teacher samples.
- `eval_teacher_pems08.py`: Evaluates the DiffSTG teacher on the PEMS08 test split.
- `eval_drift_teacher_student.py`: Evaluates the one-step student on the PEMS08 test split.
- `generate_fresh_pems08_samples.py`: Generates fresh test samples from teacher or student checkpoints and stores them as shards.
- `eval_fresh_distribution_comparison.py`: Compares fresh samples against the Teacher100 reference using distribution preservation metrics.
- `algorithm/diffstg/`: DiffSTG diffusion model and UGnet backbone implementation.
- `algorithm/drifting/student.py`: One-step student generator.
- `algorithm/drifting/losses.py`: Antisymmetric drifting, diversity matching, and energy-distance losses.
- `scripts/`: Shell scripts for the main experiment runs.

## Requirements

Python 3.9 or newer is recommended.

Main packages:

```text
torch
numpy
scipy
pandas
matplotlib
```

`nni` is optional. It is used for compatibility with the original DiffSTG training code, but the current code provides a fallback so the main training script can run without it.

Example installation:

```bash
pip install torch numpy scipy pandas matplotlib
```

## Training

### 1. Train the DiffSTG Teacher

```bash
python train.py \
  --data PEMS08 \
  --epsilon_theta UGnet \
  --N 100 \
  --sample_steps 100 \
  --ss ddpm \
  --T_h 12 \
  --T_p 12 \
  --n_samples 8 \
  --batch_size 8 \
  --is_test False \
  --teacher_checkpoint_path outputs/pems08_teacher/checkpoints/best.pt
```

Equivalent script:

```bash
bash scripts/train_teacher_pems08.sh
```

Expected checkpoint:

```text
outputs/pems08_teacher/checkpoints/best.pt
```

### 2. Generate Teacher Cache

The student is trained from precomputed teacher samples instead of calling the teacher online during student training.

```bash
python generate_teacher_cache.py \
  --checkpoint outputs/pems08_teacher/checkpoints/best.pt \
  --output_root outputs/pems08_teacher_cache_s32 \
  --splits train val \
  --num_teacher_samples 32 \
  --sample_steps 100 \
  --batch_size 16
```

Equivalent script:

```bash
NUM_TEACHER_SAMPLES=32 \
OUTPUT_ROOT=outputs/pems08_teacher_cache_s32 \
bash scripts/generate_teacher_cache_pems08.sh
```

Cache layout:

```text
outputs/pems08_teacher_cache_s32/
  metadata.json
  train/
    index.json
    shard_00000.pt
    ...
  val/
    index.json
    shard_00000.pt
    ...
```

### 3. Train the One-step Student

Train the student with the antisymmetric drifting loss:

```bash
python train_drift_teacher_student.py \
  --cache_dir outputs/pems08_teacher_cache_s32 \
  --output_dir outputs/pems08_student_drift_teacher_s32_antisymmetric \
  --epochs 50 \
  --batch_size 16 \
  --lr 1e-4 \
  --num_student_samples 8 \
  --eta 0.1 \
  --loss_type antisymmetric
```

Expected checkpoint:

```text
outputs/pems08_student_drift_teacher_s32_antisymmetric/checkpoints/best.pt
```

### 4. Fine-tune with Diversity Loss

Starting from the antisymmetric student checkpoint, fine-tune with the energy-distance auxiliary loss:

```bash
python train_drift_teacher_student.py \
  --cache_dir outputs/pems08_teacher_cache_s32 \
  --output_dir outputs/pems08_student_energy_finetune_s32_full \
  --init_checkpoint outputs/pems08_student_drift_teacher_s32_antisymmetric/checkpoints/best.pt \
  --epochs 20 \
  --batch_size 16 \
  --lr 5e-5 \
  --num_student_samples 8 \
  --eta 0.1 \
  --loss_type energy \

```

Equivalent script:

```bash
bash scripts/finetune_student_pems08_s32_energy.sh
```

Expected checkpoint:

```text
outputs/pems08_student_energy_finetune_s32_full/checkpoints/best.pt
```

## Inference and Evaluation

### Evaluate the Teacher

```bash
python eval_teacher_pems08.py \
  --checkpoint outputs/pems08_teacher/checkpoints/best.pt \
  --sample_steps 100 \
  --num_samples 8 \
  --batch_size 64 \
  --output_dir outputs/pems08_results
```

To evaluate a 40-step teacher:

```bash
python eval_teacher_pems08.py \
  --checkpoint outputs/pems08_teacher/checkpoints/best.pt \
  --sample_steps 40 \
  --num_samples 8 \
  --batch_size 64 \
  --output_dir outputs/pems08_results
```

### Evaluate the Student

```bash
python eval_drift_teacher_student.py \
  --checkpoint outputs/pems08_student_energy_finetune_s32_full/checkpoints/best.pt \
  --num_samples 8 \
  --batch_size 64 \
  --output_dir outputs/pems08_student_energy_finetune_s32_full/results
```

### Generate Fresh Samples

Distribution comparison is performed after generating fresh samples from each model on the same test conditions.

Teacher100 reference:

```bash
python generate_fresh_pems08_samples.py \
  --model_type teacher \
  --model_name Teacher100 \
  --checkpoint outputs/pems08_teacher/checkpoints/best.pt \
  --output_dir outputs/pems08_eval_fresh_refs/teacher100_steps100_samples32 \
  --sample_steps 100 \
  --num_samples 32 \
  --batch_size 32
```

Teacher40:

```bash
python generate_fresh_pems08_samples.py \
  --model_type teacher \
  --model_name Teacher40 \
  --checkpoint outputs/pems08_teacher/checkpoints/best.pt \
  --output_dir outputs/pems08_eval_fresh_refs/teacher40_steps40_samples32 \
  --sample_steps 40 \
  --num_samples 32 \
  --batch_size 32
```

Student:

```bash
python generate_fresh_pems08_samples.py \
  --model_type student \
  --model_name Student_cache32_energy \
  --checkpoint outputs/pems08_student_energy_finetune_s32_full/checkpoints/best.pt \
  --output_dir outputs/pems08_eval_fresh_refs/student_cache32_energy_samples32 \
  --sample_steps 1 \
  --num_samples 32 \
  --batch_size 32
```

### Distribution Comparison

```bash
python eval_fresh_distribution_comparison.py \
  --teacher100_dir outputs/pems08_eval_fresh_refs/teacher100_steps100_samples32 \
  --teacher40_dir outputs/pems08_eval_fresh_refs/teacher40_steps40_samples32 \
  --student_cache8_dir outputs/pems08_eval_fresh_refs/student_cache8_samples32 \
  --student_cache32_dir outputs/pems08_eval_fresh_refs/student_cache32_samples32 \
  --student_cache32_energy_dir outputs/pems08_eval_fresh_refs/student_cache32_energy_samples32 \
  --output_csv outputs/pems08_results/fresh_distribution_comparison_samples32.csv \
  --output_json outputs/pems08_results/fresh_distribution_comparison_samples32.json
```

## Metrics

- `MAE`: Masked mean absolute error between the prediction sample mean and the ground truth.
- `RMSE`: Masked root mean squared error between the prediction sample mean and the ground truth.
- `CRPS`: Empirical sample CRPS. It measures probabilistic forecast quality by considering both calibration and sharpness. Lower is better.
- `sample_diversity`: Mean pairwise absolute difference among samples generated for the same condition. Higher values indicate more diverse samples.
- `teacher100_diversity`: Sample diversity of the 100-step teacher reference.
- `relative_diversity_vs_teacher100`: Model diversity divided by Teacher100 diversity. Values close to 1 indicate that the model preserves the teacher's sample diversity.
- `EnergyDistance_to_teacher100`: Energy distance between the model sample distribution and the Teacher100 sample distribution. Lower is closer to the Teacher100 reference.
- `NFE`: Number of Function Evaluations. Teacher100 uses 100, Teacher40 uses 40, and the one-step student uses 1.
- `inference_time_total_sec`: Total inference time on the test split.
- `speedup_vs_teacher100`: Inference speedup relative to Teacher100.

## Main Results

The table below reports distribution comparison results with 32 fresh test samples per condition.

| Model | NFE | CRPS | MAE | RMSE | Diversity | Rel. Diversity | Energy Dist. | Time (sec) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Teacher100 | 100 | 40.5405 | 49.4717 | 63.9553 | 22.6510 | 1.0000 | 0.0000 | 12810.0681 | 1.00x |
| Teacher40 | 40 | 41.4183 | 50.2213 | 64.7808 | 22.0620 | 0.9740 | 0.0487 | 5129.7562 | 2.50x |
| Student_cache8 | 1 | 48.1300 | 49.0916 | 63.6964 | 1.7183 | 0.0759 | 8.3429 | 127.4411 | 100.52x |
| Student_cache32 | 1 | 47.7246 | 48.6932 | 63.2734 | 1.7410 | 0.0769 | 8.4301 | 127.5198 | 100.46x |
| Student_cache32_energy | 1 | 40.8306 | 49.3500 | 63.8051 | 22.7646 | 1.0050 | 1.4232 | 127.4384 | 100.52x |

Key observations:

- The one-step student performs inference with `NFE=1`, giving roughly `100x` speedup over Teacher100.
- The antisymmetric student preserves point forecast accuracy reasonably well, but its sample diversity is much lower.
- The energy-finetuned `Student_cache32_energy` recovers diversity close to Teacher100 while keeping CRPS near the Teacher100 result.

## Checkpoint and Output Paths

Recommended checkpoint paths:

```text
outputs/
  pems08_teacher/
    checkpoints/best.pt
  pems08_student_drift_teacher_s8_antisymmetric/
    checkpoints/best.pt
  pems08_student_drift_teacher_s32_antisymmetric/
    checkpoints/best.pt
  pems08_student_energy_finetune_s32_full/
    checkpoints/best.pt
```

Teacher cache paths:

```text
outputs/pems08_teacher_cache/
outputs/pems08_teacher_cache_s32/
```

Fresh sample paths:

```text
outputs/pems08_eval_fresh_refs/
  teacher100_steps100_samples32/
  teacher40_steps40_samples32/
  student_cache8_samples32/
  student_cache32_samples32/
  student_cache32_energy_samples32/
```

Evaluation result paths:

```text
outputs/pems08_results/
  comparison.csv
  comparison.json
  fresh_distribution_comparison_samples32.csv
  fresh_distribution_comparison_samples32.json
```

To reduce the size of the submission zip file, `outputs/pems08_teacher_cache*` and `outputs/pems08_eval_fresh_refs` can be omitted because they can be regenerated from the code and checkpoints. We recommend keeping the checkpoints and `outputs/pems08_results` for result verification.

## Reproduction Steps

To reproduce the full experiment from scratch:

1. Place the PEMS08 files at `data/dataset/PEMS08/flow.npy` and `data/dataset/PEMS08/adj.npy`.
2. Train the teacher.

```bash
bash scripts/train_teacher_pems08.sh
```

3. Generate the teacher cache.

```bash
NUM_TEACHER_SAMPLES=32 \
OUTPUT_ROOT=outputs/pems08_teacher_cache_s32 \
bash scripts/generate_teacher_cache_pems08.sh
```

4. Train the antisymmetric student.

```bash
python train_drift_teacher_student.py \
  --cache_dir outputs/pems08_teacher_cache_s32 \
  --output_dir outputs/pems08_student_drift_teacher_s32_antisymmetric \
  --epochs 50 \
  --batch_size 16 \
  --lr 1e-4 \
  --num_student_samples 8 \
  --eta 0.1 \
  --loss_type antisymmetric
```

5. Fine-tune the student with energy loss.

```bash
bash scripts/finetune_student_pems08_s32_energy.sh
```

6. Generate fresh samples.

```bash
bash scripts/generate_fresh_teacher100_ref_pems08.sh
bash scripts/generate_fresh_teacher40_pems08.sh
bash scripts/generate_fresh_student_cache8_pems08.sh
bash scripts/generate_fresh_student_cache32_pems08.sh
bash scripts/generate_fresh_student_cache32_energy_pems08.sh
```

7. Run the final distribution comparison.

```bash
bash scripts/eval_fresh_distribution_comparison_pems08.sh
```

If checkpoints are already provided, steps 2-5 can be skipped. In that case, run checkpoint-based evaluation or fresh sample generation directly.

## Notes

- The shell scripts use `${PYTHON:-python}`. Set `PYTHON=/path/to/python` if you want to run them with a specific conda or virtualenv interpreter.
- The default random seed is `2022`.
- The code uses CUDA when a GPU is available and falls back to CPU otherwise. Full teacher sampling is expensive, so GPU execution is recommended.

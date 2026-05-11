# -*- coding: utf-8 -*-
"""Generate DiffSTG codebase documentation PDF using ReportLab."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Preformatted
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

OUTPUT_PATH = "/home/ubuntu/workspace/DiffSTG/DiffSTG_코드설명.pdf"

# ── Korean/CJK font registration ────────────────────────────────────────────
_NANUM_DIR = "/usr/share/fonts/truetype/nanum"
_NOTO_DIR = "/usr/share/fonts/opentype/noto"

_FONT_CANDIDATES = {
    "regular": [
        os.path.join(_NANUM_DIR, "NanumGothic.ttf"),
        os.path.join(_NOTO_DIR, "NotoSansCJK-Regular.ttc"),
    ],
    "bold": [
        os.path.join(_NANUM_DIR, "NanumGothicBold.ttf"),
        os.path.join(_NOTO_DIR, "NotoSansCJK-Bold.ttc"),
    ],
    "mono": [
        os.path.join(_NANUM_DIR, "NanumGothicCoding.ttf"),
        os.path.join(_NANUM_DIR, "NanumGothic.ttf"),
        os.path.join(_NOTO_DIR, "NotoSansCJK-Regular.ttc"),
    ],
}

def _find_font(candidates):
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None

def _register_korean_fonts():
    regular = _find_font(_FONT_CANDIDATES["regular"])
    bold    = _find_font(_FONT_CANDIDATES["bold"])
    mono    = _find_font(_FONT_CANDIDATES["mono"])
    if regular is None or bold is None:
        raise RuntimeError(
            "Korean/CJK fonts not found. Install them with:\n"
            "  sudo apt-get install -y fonts-nanum fonts-noto-cjk"
        )
    pdfmetrics.registerFont(TTFont("Korean",      regular))
    pdfmetrics.registerFont(TTFont("Korean-Bold", bold))
    pdfmetrics.registerFont(TTFont("KoreanMono",  mono or regular))
    pdfmetrics.registerFontFamily(
        "Korean",
        normal="Korean",
        bold="Korean-Bold",
        italic="Korean",
        boldItalic="Korean-Bold",
    )

_register_korean_fonts()

def build_styles():
    base = getSampleStyleSheet()

    title = ParagraphStyle(
        "DocTitle",
        parent=base["Title"],
        fontName="Korean-Bold",
        fontSize=22,
        leading=28,
        spaceAfter=14,
        textColor=colors.HexColor("#1a237e"),
        alignment=TA_CENTER,
    )
    subtitle = ParagraphStyle(
        "DocSubtitle",
        parent=base["Normal"],
        fontName="Korean",
        fontSize=12,
        leading=16,
        spaceAfter=6,
        textColor=colors.HexColor("#455a64"),
        alignment=TA_CENTER,
    )
    h1 = ParagraphStyle(
        "H1",
        parent=base["Heading1"],
        fontName="Korean-Bold",
        fontSize=16,
        leading=20,
        spaceBefore=18,
        spaceAfter=8,
        textColor=colors.HexColor("#1565c0"),
        borderPad=4,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=base["Heading2"],
        fontName="Korean-Bold",
        fontSize=13,
        leading=17,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor("#0277bd"),
    )
    h3 = ParagraphStyle(
        "H3",
        parent=base["Heading3"],
        fontName="Korean-Bold",
        fontSize=11,
        leading=15,
        spaceBefore=8,
        spaceAfter=4,
        textColor=colors.HexColor("#00838f"),
    )
    body = ParagraphStyle(
        "Body",
        parent=base["Normal"],
        fontName="Korean",
        fontSize=10,
        leading=15,
        spaceAfter=5,
        alignment=TA_JUSTIFY,
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=base["Normal"],
        fontName="Korean",
        fontSize=10,
        leading=14,
        spaceAfter=3,
        leftIndent=18,
        bulletIndent=6,
    )
    code = ParagraphStyle(
        "Code",
        parent=base["Code"],
        fontName="KoreanMono",
        fontSize=8,
        leading=11,
        spaceAfter=4,
        backColor=colors.HexColor("#f5f5f5"),
        leftIndent=12,
        rightIndent=12,
        borderColor=colors.HexColor("#cccccc"),
        borderWidth=0.5,
        borderPad=6,
    )
    note = ParagraphStyle(
        "Note",
        parent=base["Normal"],
        fontName="Korean",
        fontSize=9,
        leading=13,
        spaceAfter=4,
        leftIndent=12,
        textColor=colors.HexColor("#546e7a"),
        borderColor=colors.HexColor("#b0bec5"),
        borderWidth=0.5,
        borderPad=5,
        backColor=colors.HexColor("#eceff1"),
    )
    return dict(title=title, subtitle=subtitle, h1=h1, h2=h2, h3=h3,
                body=body, bullet=bullet, code=code, note=note)


def hr(styles):
    return HRFlowable(width="100%", thickness=1, color=colors.HexColor("#90caf9"), spaceAfter=6)


def section(title_text, style_h, content_blocks):
    return [Paragraph(title_text, style_h)] + content_blocks


def build_document():
    S = build_styles()

    story = []

    # ── Title page ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("DiffSTG 프로젝트", S["title"]))
    story.append(Paragraph("코드 및 파일 구조 설명서", S["subtitle"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(hr(S))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "SIGSPATIAL 2023 논문 「DiffSTG: Probabilistic Spatio-Temporal Graph Forecasting "
        "with Denoising Diffusion Models」의 PyTorch 구현 및 확장 코드 문서",
        S["subtitle"],
    ))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("작성일: 2026-05-11", S["subtitle"]))
    story.append(PageBreak())

    # ── 1. 프로젝트 개요 ────────────────────────────────────────────────────────
    story.append(Paragraph("1. 프로젝트 개요", S["h1"]))
    story.append(hr(S))
    story.append(Paragraph(
        "DiffSTG는 <b>확산 모델(Denoising Diffusion Probabilistic Model, DDPM)</b>을 이용하여 "
        "교통·공기질 등 시공간 그래프 데이터를 <b>확률론적으로 예측</b>하는 프레임워크입니다. "
        "과거 T_h 타임스텝의 관측값을 조건(condition)으로 받아 미래 T_p 타임스텝의 "
        "<b>다중 샘플(probabilistic forecast)</b>을 생성합니다.",
        S["body"],
    ))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("주요 특징:", S["h3"]))
    for item in [
        "• 시간(Temporal) 특징 추출: 팽창 인과 합성곱(Dilated Causal Conv, TCN)",
        "• 공간(Spatial) 특징 추출: 비대칭 GCN (Asymmetric Graph Convolution)",
        "• 백본 아키텍처: U-Net 구조의 UGnet",
        "• 샘플링 전략: DDPM 또는 DDIM (가속 샘플링)",
        "• 확장 기능: Drift-to-Teacher 지식 증류 (1-step 학생 모델)",
    ]:
        story.append(Paragraph(item, S["bullet"]))

    story.append(Spacer(1, 0.3 * cm))

    # ── 2. 디렉토리 구조 ────────────────────────────────────────────────────────
    story.append(Paragraph("2. 디렉토리 구조", S["h1"]))
    story.append(hr(S))

    tree = """\
DiffSTG/
├── train.py                        # 교사 모델(DiffSTG) 학습 진입점
├── train_drift_teacher_student.py  # 학생 모델 학습 (Drift-to-Teacher)
├── generate_teacher_cache.py       # 교사 샘플 캐시 생성
├── eval_teacher_pems08.py          # 교사 모델 평가 (PEMS08)
├── eval_drift_teacher_student.py   # 학생 모델 평가
├── easydict.py                     # EasyDict 유틸리티
│
├── algorithm/
│   ├── dataset.py                  # 데이터셋 클래스 (CleanDataset, TrafficDataset)
│   ├── diffstg/
│   │   ├── model.py                # DiffSTG 확산 모델 핵심 구현
│   │   ├── ugnet.py                # UGnet 백본 (U-Net + GCN)
│   │   ├── graph_algo.py           # 그래프 알고리즘 (라플라시안, 랜덤워크 등)
│   │   └── checkpoint.py           # 체크포인트 저장/로드
│   └── drifting/
│       ├── student.py              # DriftTeacherStudent (1-step 생성기)
│       └── losses.py               # RBF Drift-to-Teacher 손실 함수
│
├── utils/
│   ├── eval.py                     # 평가 지표 (MAE, RMSE, CRPS, MIS)
│   ├── common_utils.py             # 공통 유틸리티
│   └── gpu_dispatch.py             # GPU 자동 선택
│
├── scripts/                        # 실험 실행 쉘 스크립트
├── data/dataset/
│   ├── PEMS08/                     # PEMS08 교통 데이터 (170 노드)
│   └── AIR_GZ/                     # 광저우 공기질 데이터 (41 노드)
└── outputs/                        # 실험 결과, 모델 체크포인트, 캐시"""

    story.append(Preformatted(tree, S["code"]))
    story.append(PageBreak())

    # ── 3. 핵심 모델: DiffSTG ───────────────────────────────────────────────────
    story.append(Paragraph("3. 핵심 모델: DiffSTG (algorithm/diffstg/model.py)", S["h1"]))
    story.append(hr(S))
    story.append(Paragraph(
        "DiffSTG는 <b>마스크 확산 모델</b>로, 미래 구간을 0으로 마스킹한 입력을 조건(c)으로 "
        "받아 노이즈 제거 과정을 통해 미래를 예측합니다.",
        S["body"],
    ))

    story.append(Paragraph("3.1 핵심 수식", S["h2"]))
    story.append(Paragraph(
        "순방향 과정 (Forward Process): 가우시안 노이즈를 점진적으로 추가합니다.",
        S["body"],
    ))
    story.append(Preformatted(
        "q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)\n"
        "  beta: 노이즈 스케줄 (uniform 또는 quad)\n"
        "  alpha_bar_t = prod(1 - beta_i, i=1..t)",
        S["code"],
    ))
    story.append(Paragraph(
        "역방향 과정 (Reverse Process): UGnet이 노이즈 eps_theta를 예측합니다.",
        S["body"],
    ))
    story.append(Preformatted(
        "p(x_{t-1} | x_t, c) -> UGnet이 eps_theta(x_t, t, c) 추정\n"
        "mean = (1/sqrt(alpha_t)) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_theta)\n"
        "Loss = MSE(eps, eps_theta(x_t, t, c))",
        S["code"],
    ))

    story.append(Paragraph("3.2 주요 메서드", S["h2"]))
    methods = [
        ["메서드", "설명"],
        ["q_xt_x0(x0, t, eps)", "순방향: x_0에서 x_t 샘플링"],
        ["p_sample(xt, t, c)", "역방향: DDPM 한 스텝 노이즈 제거"],
        ["p_sample_loop(c)", "DDPM 전체 역방향 루프 (N→0)"],
        ["p_sample_loop_ddim(c)", "DDIM 가속 샘플링 (skip timesteps)"],
        ["evaluate(input, n_samples)", "다중 샘플 예측 (ddpm/ddim_multi/ddim_one)"],
        ["loss(x0, c)", "MSE 훈련 손실 계산"],
    ]
    t = Table(methods, colWidths=[7 * cm, 10 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565c0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Korean-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Korean"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#e3f2fd")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#90caf9")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("3.3 DDIM 가속 샘플링 (generalized_steps)", S["h2"]))
    story.append(Paragraph(
        "generalized_steps 함수는 전체 N개 타임스텝 중 일부만 건너뛰며 "
        "역방향 확산을 수행합니다. quad 또는 uniform 스케줄로 시퀀스를 선택하고, "
        "eta 파라미터로 결정론적(eta=0)~확률적(eta=1) 샘플링을 조절합니다.",
        S["body"],
    ))
    story.append(PageBreak())

    # ── 4. UGnet ────────────────────────────────────────────────────────────────
    story.append(Paragraph("4. UGnet 백본 (algorithm/diffstg/ugnet.py)", S["h1"]))
    story.append(hr(S))
    story.append(Paragraph(
        "UGnet은 <b>U-Net + Graph Convolution Network</b>를 결합한 시공간 신경망입니다. "
        "확산 타임스텝 t의 임베딩을 각 ResidualBlock에 주입하여 노이즈 수준을 인식합니다.",
        S["body"],
    ))

    story.append(Paragraph("4.1 구성 블록", S["h2"]))
    blocks = [
        ["블록", "입력 → 출력", "역할"],
        ["TimeEmbedding", "timestep t → (B, d_h)", "Sinusoidal 타임스텝 임베딩"],
        ["TcnBlock", "(B,C,V,T) → (B,C,V,T)", "팽창 인과 합성곱 + 잔차 연결"],
        ["SpatialBlock", "(B,C,V,T) → (B,C,V,T)", "비대칭 그래프 합성곱 (GCN)"],
        ["ResidualBlock", "(B,C_in,V,T) → (B,C_out,V,T)", "TCN + 시간임베딩 + GCN + LayerNorm"],
        ["DownBlock / Downsample", "(B,C,V,T) → (B,C,V,T/2)", "인코더 + 시간 해상도 감소"],
        ["UpBlock / Upsample", "(B,C,V,T) → (B,C,V,2T)", "디코더 + skip connection"],
        ["MiddleBlock", "(B,C,V,T) → (B,C,V,T)", "U-Net 병목 (2개의 ResidualBlock)"],
    ]
    t = Table(blocks, colWidths=[4 * cm, 5 * cm, 8 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0277bd")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Korean-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Korean"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#e1f5fe")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#81d4fa")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("4.2 UGnet 순전파 흐름", S["h2"]))
    story.append(Preformatted(
        "입력:\n"
        "  x   : (B, F, V, T)     # 현재 확산 스텝의 노이즈 텐서\n"
        "  t   : (B,)             # 확산 타임스텝 인덱스\n"
        "  c   : (x_masked, pos_w, pos_d)\n\n"
        "1. x_cat = cat(x, x_masked, dim=3)   → (B, F, V, 2T)\n"
        "2. x     = x_proj(x_cat)              → (B, d_h, V, 2T)\n"
        "3. t_emb = TimeEmbedding(t, d_h)      → (B, d_h)\n"
        "4. 인코더: Down0 → Down1 → ... → Downsample → ...  (skip 저장)\n"
        "5. 병목:   MiddleBlock\n"
        "6. 디코더: Upsample → Up0(+skip) → Up1(+skip) → ...\n"
        "7. 출력:   Conv2d(d_h→F) + Linear(2T→T)         → (B, F, V, T)",
        S["code"],
    ))

    story.append(Paragraph("4.3 그래프 지원 행렬", S["h2"]))
    story.append(Paragraph(
        "인접 행렬 A로부터 <b>순방향(a1)</b>과 <b>역방향(a2)</b> 비대칭 정규화 행렬을 생성하여 "
        "양방향 공간 정보를 포착합니다. supports_len=2.",
        S["body"],
    ))
    story.append(Preformatted(
        "a1 = asym_adj(A)         # D^{-1} A\n"
        "a2 = asym_adj(A^T)       # D^{-1} A^T\n"
        "supports = stack([a1, a2])  # (2, V, V)",
        S["code"],
    ))
    story.append(PageBreak())

    # ── 5. 데이터셋 ─────────────────────────────────────────────────────────────
    story.append(Paragraph("5. 데이터셋 처리 (algorithm/dataset.py)", S["h1"]))
    story.append(hr(S))

    story.append(Paragraph("5.1 CleanDataset", S["h2"]))
    story.append(Paragraph(
        "원시 데이터를 로드하고 훈련 세트 평균·표준편차로 Z-score 정규화합니다. "
        "지원 데이터셋: PEMS08 (교통), AIR_GZ (공기질), AIR_BJ, Metro.",
        S["body"],
    ))
    story.append(Preformatted(
        "PEMS08: 170개 센서, 5분 간격 교통량, 총 17,856 타임스텝\n"
        "  train: 0 ~ 60%  |  val: 60~80%  |  test: 80~100%\n\n"
        "정규화: feature = (feature - mean_train) / std_train",
        S["code"],
    ))

    story.append(Paragraph("5.2 TrafficDataset", S["h2"]))
    story.append(Paragraph(
        "CleanDataset을 받아 슬라이딩 윈도우 방식으로 (history, label) 쌍을 생성합니다. "
        "각 샘플은 T_h개의 과거 + T_p개의 미래 시퀀스를 포함합니다.",
        S["body"],
    ))
    features = [
        ["필드", "형상", "설명"],
        ["label", "(T_p, V, D)", "정규화된 미래값 (예측 대상)"],
        ["node_feature", "(T_h, V, D)", "정규화된 과거 관측값 (조건)"],
        ["pos_w", "(T_h,)", "요일 위치 인덱스 (0~6)"],
        ["pos_d", "(T_h,)", "하루 내 위치 인덱스 (0~points_per_day)"],
    ]
    t = Table(features, colWidths=[3.5 * cm, 4 * cm, 9.5 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#00838f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Korean-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Korean"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#e0f7fa")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#80deea")),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(PageBreak())

    # ── 6. Drifting 모듈 ────────────────────────────────────────────────────────
    story.append(Paragraph("6. Drift-to-Teacher 지식 증류 (algorithm/drifting/)", S["h1"]))
    story.append(hr(S))
    story.append(Paragraph(
        "Drift-to-Teacher는 느린 교사 모델(N=100 스텝 DDIM)로부터 "
        "<b>NFE=1(단 한 번의 순전파)</b>로 동일한 품질을 내는 학생 모델을 훈련합니다. "
        "RBF 커널 기반의 인력(Attraction) 손실로 학생 분포를 교사 분포로 끌어당깁니다.",
        S["body"],
    ))

    story.append(Paragraph("6.1 DriftTeacherStudent (student.py)", S["h2"]))
    story.append(Paragraph(
        "UGnet을 백본으로 사용하는 1-step 조건부 생성기입니다. "
        "랜덤 노이즈 z_future를 받아 단 하나의 UGnet forward call로 미래를 예측합니다.",
        S["body"],
    ))
    story.append(Preformatted(
        "forward_single(z_future, x_masked, pos_w, pos_d):\n"
        "  z_full = zeros(B, F, V, T_total)\n"
        "  z_full[:, :, :, T_h:] = z_future   # 미래 구간에 노이즈 배치\n"
        "  t = zeros(B)                         # 타임스텝 0 (단 1회 호출)\n"
        "  out = backbone(z_full, t, condition)\n"
        "  return out[:, :, :, -T_p:]           # 미래 구간만 반환\n\n"
        "forward(x_masked, pos_w, pos_d, num_samples):\n"
        "  z ~ N(0, I)  shape=(B, num_samples, F, V, T_p)\n"
        "  → num_samples개 병렬 샘플 생성\n"
        "  → output: (B, num_samples, T_p, V, F)",
        S["code"],
    ))

    story.append(Paragraph("6.2 RBF Drift-to-Teacher 손실 (losses.py)", S["h2"]))
    story.append(Paragraph(
        "각 학생 샘플 y_i를 교사 샘플 {t_j}로 끌어당기는 소프트 인력을 계산합니다.",
        S["body"],
    ))
    story.append(Preformatted(
        "rbf_drift_to_teacher_loss(student_samples, teacher_samples, eta, sigma):\n\n"
        "  # RBF 커널 가중치 계산\n"
        "  dist2[i,j] = ||y_i - t_j||^2\n"
        "  sigma  = median(||y_i - t_j||)   (auto_sigma, sigma 미지정 시)\n"
        "  w[i,j] = exp(-dist2[i,j] / 2*sigma^2)\n"
        "  w[i,j] = w[i,j] / sum_j(w[i,j])  (정규화)\n\n"
        "  # Drift (인력) 벡터 계산\n"
        "  drift_i = sum_j w[i,j] * (t_j - y_i)\n\n"
        "  # Stop-gradient 타겟\n"
        "  target_i = stop_grad(y_i + eta * drift_i)\n\n"
        "  # MSE 손실\n"
        "  loss = mean_i ||y_i - target_i||^2",
        S["code"],
    ))
    story.append(Paragraph(
        "• <b>eta</b>: 드리프트 강도 (기본값 0.1). 클수록 교사에 강하게 수렴합니다.",
        S["bullet"],
    ))
    story.append(Paragraph(
        "• <b>sigma</b>: RBF 대역폭. None이면 중위 거리(median pairwise distance)를 자동 사용.",
        S["bullet"],
    ))
    story.append(PageBreak())

    # ── 7. 학습 파이프라인 ───────────────────────────────────────────────────────
    story.append(Paragraph("7. 학습 파이프라인", S["h1"]))
    story.append(hr(S))

    story.append(Paragraph("7.1 교사 모델 학습 (train.py)", S["h2"]))
    story.append(Paragraph(
        "표준 DiffSTG 확산 모델을 학습합니다. Adam 옵티마이저, ReduceLROnPlateau 스케줄러, "
        "Early Stopping을 사용하며 검증 MAE 기준으로 최적 모델을 저장합니다.",
        S["body"],
    ))
    steps = [
        ["단계", "설명"],
        ["1. 데이터 로드", "CleanDataset → TrafficDataset (train/val/test 분할)"],
        ["2. 배치 구성", "history + future → x0, history + zeros → x_masked"],
        ["3. 순방향 과정", "t ~ Uniform(0, N) 샘플, x_t = q_xt_x0(x0, t)"],
        ["4. 역방향 예측", "eps_theta = UGnet(x_t, t, c)"],
        ["5. 손실 계산", "loss = 10 * MSE(eps, eps_theta)"],
        ["6. 검증", "Val MAE 기준 best 모델 저장 (early_stop=10 epoch)"],
        ["7. 최종 평가", "DDIM 40-step, n_samples=8 로 테스트 세트 평가"],
    ]
    t = Table(steps, colWidths=[4.5 * cm, 12.5 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565c0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Korean-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Korean"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#e3f2fd")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#90caf9")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("7.2 교사 캐시 생성 (generate_teacher_cache.py)", S["h2"]))
    story.append(Paragraph(
        "학습된 교사 모델을 이용해 train/val 세트 각 샘플에 대한 예측 샘플을 미리 생성하여 "
        "디스크에 저장합니다. 학생 모델 학습 시 캐시를 재사용하므로 교사 모델을 반복 실행할 "
        "필요가 없어 학습 속도가 크게 향상됩니다.",
        S["body"],
    ))
    story.append(Preformatted(
        "저장 구조:\n"
        "outputs/pems08_teacher_cache/\n"
        "  metadata.json              # 데이터셋·모델 메타정보\n"
        "  train/\n"
        "    index.json               # 샤드 목록\n"
        "    shard_00000.pt           # {history, x_masked, gt_future,\n"
        "    shard_00001.pt           #  teacher_samples, pos_w, pos_d, adj}\n"
        "  val/\n"
        "    index.json\n"
        "    shard_00000.pt ...",
        S["code"],
    ))

    story.append(Paragraph("7.3 학생 모델 학습 (train_drift_teacher_student.py)", S["h2"]))
    story.append(Paragraph(
        "TeacherCacheDataset으로 캐시된 교사 샘플을 로드하여 "
        "DriftTeacherStudent를 Drift-to-Teacher 손실로 학습합니다.",
        S["body"],
    ))
    story.append(Preformatted(
        "python train_drift_teacher_student.py \\\n"
        "  --cache_dir outputs/pems08_teacher_cache \\\n"
        "  --output_dir outputs/pems08_student_drift_teacher \\\n"
        "  --epochs 50 --batch_size 16 --lr 1e-4 \\\n"
        "  --num_student_samples 8 --eta 0.1",
        S["code"],
    ))
    story.append(PageBreak())

    # ── 8. 평가 지표 ─────────────────────────────────────────────────────────────
    story.append(Paragraph("8. 평가 지표 (utils/eval.py)", S["h1"]))
    story.append(hr(S))
    story.append(Paragraph(
        "결정론적 지표와 확률론적 지표를 모두 계산합니다.",
        S["body"],
    ))
    metrics = [
        ["지표", "유형", "설명"],
        ["MAE", "결정론적", "평균 절대 오차 (샘플 평균으로 계산)"],
        ["RMSE", "결정론적", "평균 제곱근 오차"],
        ["MAPE", "결정론적", "평균 절대 백분율 오차 (%)"],
        ["CRPS", "확률론적", "연속 랭크 확률 점수 (분위수 손실 평균)"],
        ["MIS", "확률론적", "평균 구간 점수 (90% 신뢰 구간 기반)"],
        ["Diversity", "확률론적", "샘플 간 평균 절대 차이 (다양성 척도)"],
    ]
    t = Table(metrics, colWidths=[3 * cm, 3.5 * cm, 10.5 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a148c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Korean-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Korean"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ede7f6")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b39ddb")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("CRPS 계산 방식:", S["h3"]))
    story.append(Preformatted(
        "quantiles = [0.05, 0.10, ..., 0.95]  (19개)\n"
        "CRPS = mean_q [ quantile_loss(target, forecast_quantile_q, q) / denominator ]",
        S["code"],
    ))

    # ── 9. 실험 결과 ─────────────────────────────────────────────────────────────
    story.append(Paragraph("9. PEMS08 실험 결과", S["h1"]))
    story.append(hr(S))
    story.append(Paragraph(
        "PEMS08 테스트 세트(3,549 샘플, 170 노드, T_h=12, T_p=12)에서의 비교 결과입니다.",
        S["body"],
    ))
    results = [
        ["모델", "NFE", "MAE", "RMSE", "CRPS", "추론시간(s)", "속도향상"],
        ["DiffSTG 교사 (100-step)", "100", "49.66", "64.33", "41.61", "3120", "1×"],
        ["DiffSTG 교사 (40-step)",  "40",  "50.36", "65.07", "42.43", "1252", "2.5×"],
        ["Drift-to-Teacher 학생",   "1",   "48.86", "63.40", "48.46", "31",   "102×"],
    ]
    t = Table(results, colWidths=[5 * cm, 1.5 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 3 * cm, 2.1 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#bf360c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Korean-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Korean"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fbe9e7")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ffab91")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 3), (-1, 3), "Korean-Bold"),  # highlight student row
        ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#fff3e0")),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "※ 학생 모델은 MAE·RMSE 기준으로 100-step 교사보다 우수하면서 "
        "<b>102배 빠른 추론 속도</b>를 달성합니다.",
        S["note"],
    ))
    story.append(PageBreak())

    # ── 10. 유틸리티 ─────────────────────────────────────────────────────────────
    story.append(Paragraph("10. 유틸리티 모듈", S["h1"]))
    story.append(hr(S))

    story.append(Paragraph("10.1 graph_algo.py", S["h2"]))
    story.append(Paragraph(
        "그래프 인접 행렬 전처리 함수들을 제공합니다.",
        S["body"],
    ))
    funcs = [
        ["함수", "설명"],
        ["asym_adj(adj)", "비대칭 정규화: D^{-1} A"],
        ["sym_adj(adj)", "대칭 정규화: D^{-1/2} A D^{-1/2}"],
        ["calculate_normalized_laplacian(adj)", "정규화 라플라시안: I - D^{-1/2} A D^{-1/2}"],
        ["calculate_random_walk_matrix(adj)", "랜덤워크 행렬: D^{-1} A"],
        ["calculate_scaled_laplacian(adj)", "스케일 라플라시안 (Chebyshev 다항식용)"],
    ]
    t = Table(funcs, colWidths=[7 * cm, 10 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#00838f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Korean-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Korean"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#e0f7fa")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#80deea")),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("10.2 checkpoint.py", S["h2"]))
    story.append(Paragraph(
        "모델 가중치, 옵티마이저 상태, 에포크, 최적 검증 지표, 설정, 스케일러를 "
        "하나의 .pt 파일에 저장하고 로드합니다.",
        S["body"],
    ))

    story.append(Paragraph("10.3 common_utils.py", S["h2"]))
    for item in [
        "• gather(consts, t): 확산 타임스텝 t에 해당하는 beta/alpha 값 추출",
        "• to_device(batch, device): 배치 텐서를 GPU로 이동",
        "• Logger: 터미널 + 파일 동시 출력 로거",
        "• save2file_meta: 실험 결과를 CSV에 추가 저장",
        "• draw_predicted_distribution: 예측 분위수 구간을 시각화",
    ]:
        story.append(Paragraph(item, S["bullet"]))

    story.append(Paragraph("10.4 gpu_dispatch.py", S["h2"]))
    story.append(Paragraph(
        "사용 가능한 GPU 중 메모리 여유가 충분한 GPU를 자동으로 선택합니다.",
        S["body"],
    ))

    # ── 11. 실행 방법 ─────────────────────────────────────────────────────────────
    story.append(Paragraph("11. 실행 방법", S["h1"]))
    story.append(hr(S))

    story.append(Paragraph("Step 1: 교사 모델 학습", S["h2"]))
    story.append(Preformatted(
        "bash scripts/train_teacher_pems08.sh\n"
        "# 또는\n"
        "python train.py --data PEMS08 --N 100 --ss ddpm --n_samples 8",
        S["code"],
    ))

    story.append(Paragraph("Step 2: 교사 샘플 캐시 생성", S["h2"]))
    story.append(Preformatted(
        "bash scripts/generate_teacher_cache_pems08.sh\n"
        "# 또는\n"
        "python generate_teacher_cache.py \\\n"
        "  --checkpoint outputs/pems08_teacher/checkpoints/best.pt \\\n"
        "  --output_root outputs/pems08_teacher_cache \\\n"
        "  --num_teacher_samples 8 --sample_steps 100",
        S["code"],
    ))

    story.append(Paragraph("Step 3: 학생 모델 학습", S["h2"]))
    story.append(Preformatted(
        "bash scripts/train_student_pems08_drift_teacher.sh",
        S["code"],
    ))

    story.append(Paragraph("Step 4: 평가", S["h2"]))
    story.append(Preformatted(
        "# 교사 모델 평가\n"
        "bash scripts/eval_teacher_pems08.sh\n\n"
        "# 학생 모델 평가\n"
        "bash scripts/eval_student_pems08_drift_teacher.sh\n"
        "# 또는\n"
        "python eval_drift_teacher_student.py \\\n"
        "  --checkpoint outputs/pems08_student_drift_teacher/checkpoints/best.pt \\\n"
        "  --num_samples 8",
        S["code"],
    ))

    # ── 12. 주요 하이퍼파라미터 ───────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("12. 주요 하이퍼파라미터", S["h1"]))
    story.append(hr(S))

    params = [
        ["파라미터", "기본값", "설명"],
        ["N", "100 / 200", "순방향 확산 총 스텝 수"],
        ["sample_steps", "100", "샘플링 시 실제 사용 스텝 수 (≤ N)"],
        ["beta_schedule", "quad", "노이즈 스케줄 (quad: 이차 / uniform: 균등)"],
        ["beta_end", "0.02 ~ 0.1", "최대 노이즈 분산"],
        ["d_h", "32", "UGnet 기본 채널 수 (hidden dimension)"],
        ["T_h", "12", "과거 입력 타임스텝 수"],
        ["T_p", "12", "예측 타임스텝 수"],
        ["n_samples", "8", "평가 시 샘플 수"],
        ["channel_multipliers", "[1, 2]", "U-Net 해상도별 채널 배수"],
        ["eta", "0.1", "Drift-to-Teacher 인력 강도"],
        ["sigma", "auto", "RBF 커널 대역폭 (None=중위 거리 자동)"],
        ["batch_size", "32 (교사) / 16 (학생)", "미니배치 크기"],
        ["lr", "0.002 (교사) / 1e-4 (학생)", "학습률"],
        ["early_stop", "10", "Early Stopping 인내 에포크"],
    ]
    t = Table(params, colWidths=[5 * cm, 4 * cm, 8 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#37474f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Korean-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Korean"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eceff1")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#90a4ae")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    # ── 마무리 ────────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    story.append(hr(S))
    story.append(Paragraph(
        "참고 논문: Wen et al., \"DiffSTG: Probabilistic Spatio-Temporal Graph Forecasting "
        "with Denoising Diffusion Models\", SIGSPATIAL 2023. arXiv:2301.13629",
        S["note"],
    ))

    return story


def main():
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
        title="DiffSTG 코드 설명서",
        author="Claude",
        subject="DiffSTG Codebase Documentation",
    )
    story = build_document()
    doc.build(story)
    print(f"PDF saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

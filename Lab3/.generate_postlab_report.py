from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "POSTLAB_REPORT.docx"
NAVY = "17365D"
BLUE = "2F75B5"
PALE_BLUE = "D9EAF7"
LIGHT_GRAY = "F2F4F7"
DARK = "253247"


def shade(cell, fill):
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)


def set_cell_text(cell, text, *, bold=False, color=DARK, size=8, align=None):
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    if align is not None:
        paragraph.alignment = align
    run = paragraph.add_run(str(text))
    run.bold = bold
    run.font.name = "Aptos"
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_repeat_table_header(row):
    properties = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    properties.append(repeat)


def add_table(doc, headers, rows, widths, font_size=8):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = False
    for i, width in enumerate(widths):
        table.columns[i].width = Inches(width)

    header = table.rows[0]
    set_repeat_table_header(header)
    for i, label in enumerate(headers):
        header.cells[i].width = Inches(widths[i])
        set_cell_text(
            header.cells[i],
            label,
            bold=True,
            color="FFFFFF",
            size=font_size,
            align=WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER,
        )
        shade(header.cells[i], NAVY)

    for row_index, values in enumerate(rows):
        cells = table.add_row().cells
        for i, value in enumerate(values):
            cells[i].width = Inches(widths[i])
            set_cell_text(
                cells[i],
                value,
                size=font_size,
                align=WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER,
            )
            if row_index % 2 == 1:
                shade(cells[i], LIGHT_GRAY)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_caption(doc, text):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(9)
    run = paragraph.add_run(text)
    run.italic = True
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor.from_string("526173")


def add_figure(doc, filename, caption, width=6.9):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(1)
    paragraph.add_run().add_picture(str(ROOT / "plots" / filename), width=Inches(width))
    add_caption(doc, caption)


def add_bullet(doc, text):
    paragraph = doc.add_paragraph(style="List Bullet")
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.add_run(text)
    return paragraph


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instruction)
    run._r.append(end)


doc = Document()
section = doc.sections[0]
section.top_margin = Inches(0.65)
section.bottom_margin = Inches(0.65)
section.left_margin = Inches(0.72)
section.right_margin = Inches(0.72)

normal = doc.styles["Normal"]
normal.font.name = "Aptos"
normal.font.size = Pt(10)
normal.font.color.rgb = RGBColor.from_string(DARK)
normal.paragraph_format.space_after = Pt(5)
normal.paragraph_format.line_spacing = 1.08

for style_name, size in (("Title", 25), ("Heading 1", 17), ("Heading 2", 12), ("Heading 3", 10)):
    style = doc.styles[style_name]
    style.font.name = "Aptos Display" if style_name in ("Title", "Heading 1") else "Aptos"
    style.font.size = Pt(size)
    style.font.bold = style_name != "Title" or True
    style.font.color.rgb = RGBColor.from_string(NAVY if style_name != "Heading 2" else BLUE)
    style.paragraph_format.space_before = Pt(10 if style_name != "Title" else 0)
    style.paragraph_format.space_after = Pt(5)

header = section.header.paragraphs[0]
header.text = "ELEC 475  |  LAB 3 POST-LAB"
header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
for run in header.runs:
    run.font.name = "Aptos"
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor.from_string("6B7785")
add_page_number(section.footer.paragraphs[0])

# Cover page
for _ in range(4):
    doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("ELEC 475")
run.bold = True
run.font.size = Pt(16)
run.font.color.rgb = RGBColor.from_string(BLUE)

p = doc.add_paragraph(style="Title")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run("Lab 3 — Post-Lab Report")

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Pet Nose Localization Correction")
run.font.size = Pt(17)
run.font.color.rgb = RGBColor.from_string(DARK)

doc.add_paragraph()
for label, value in (
    ("Student", "Lucas Balog"),
    ("Student number", "20320245"),
    ("Report date", "November 5, 2026"),
):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(f"{label}: ")
    run.bold = True
    run.font.color.rgb = RGBColor.from_string(NAVY)
    p.add_run(value)

doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.left_indent = Inches(0.8)
p.paragraph_format.right_indent = Inches(0.8)
run = p.add_run(
    "This report describes a local patch-based correction stage for pet nose "
    "localization and evaluates its effect on the Lab 3 test predictions."
)
run.italic = True
run.font.size = Pt(11)
run.font.color.rgb = RGBColor.from_string("526173")
doc.add_page_break()

# Introduction
doc.add_heading("Introduction", level=1)
doc.add_paragraph(
    "The post-lab adds a second-stage localizer to the whole-image predictions "
    "from SnoutNet, SnoutNet-A, and SnoutNet-V. The correction model receives a "
    "64 × 64 image patch centered on the first-stage nose prediction and returns "
    "a two-dimensional offset. Its corrected coordinate is the rounded patch "
    "center plus 64 times the predicted normalized offset. Training uses the "
    "training split only; the reported accuracy is computed on the held-out test "
    "predictions."
)

# Model
doc.add_heading("1. Model", level=1)
doc.add_heading("1.1 CorrectionNet", level=2)
doc.add_paragraph(
    "CorrectionNet maps an ImageNet-normalized RGB patch of shape (3, 64, 64) "
    "to a two-element normalized offset (du/64, dv/64). Each convolution block "
    "uses a 3 × 3 convolution with padding 1, batch normalization, ReLU, and "
    "2 × 2 max pooling with stride 2. The spatial dimensions halve at each block."
)
add_table(
    doc,
    ["Stage", "Operations", "Output shape"],
    [
        ("Input", "RGB patch", "(3, 64, 64)"),
        ("Block 1", "Conv 3→32; BN; ReLU; max-pool 2×2", "(32, 32, 32)"),
        ("Block 2", "Conv 32→64; BN; ReLU; max-pool 2×2", "(64, 16, 16)"),
        ("Block 3", "Conv 64→128; BN; ReLU; max-pool 2×2", "(128, 8, 8)"),
        ("Block 4", "Conv 128→128; BN; ReLU; max-pool 2×2", "(128, 4, 4)"),
        ("Head", "Flatten 2,048; Linear 2,048→256; ReLU; Linear 256→2", "(2,)"),
    ],
    [0.85, 4.7, 1.25],
    font_size=8,
)
doc.add_paragraph(
    "The model has 766,594 trainable parameters. Its output is not constrained "
    "by a final activation; inference converts the two values into pixel offsets "
    "by multiplying by 64."
)

doc.add_heading("1.2 CorrectionNetModified", level=2)
doc.add_paragraph(
    "CorrectionNetModified preserves the same input, four downsampling stages, "
    "flattened feature size, and two-coordinate output. It replaces each basic "
    "convolution block with a residual block containing two 3 × 3 convolutions. "
    "Each convolution is followed by batch normalization; the first also has a "
    "ReLU. The block adds the input (or a learned 1 × 1 projection when the "
    "channel count changes), applies ReLU, and then max-pools by two. An identity "
    "skip is used when channel counts already match. Its fully connected head is "
    "2,048→256→2, with ReLU and dropout (p=0.25) between the linear layers."
)
add_table(
    doc,
    ["Stage", "Residual block / head", "Output shape"],
    [
        ("Block 1", "3→32; two 3×3 convs; 1×1 projection skip; add; ReLU; pool", "(32, 32, 32)"),
        ("Block 2", "32→64; two 3×3 convs; 1×1 projection skip; add; ReLU; pool", "(64, 16, 16)"),
        ("Block 3", "64→128; two 3×3 convs; 1×1 projection skip; add; ReLU; pool", "(128, 8, 8)"),
        ("Block 4", "128→128; two 3×3 convs; identity skip; add; ReLU; pool", "(128, 4, 4)"),
        ("Head", "Flatten 2,048; Linear→256; ReLU; Dropout 0.25; Linear→2", "(2,)"),
    ],
    [0.85, 4.7, 1.25],
    font_size=8,
)
doc.add_paragraph(
    "CorrectionNetModified has 1,119,202 trainable parameters. Neither model "
    "uses pretrained correction weights."
)

# Training
doc.add_heading("2. Training", level=1)
doc.add_paragraph(
    "Both correction architectures were trained on the 5,494 labeled training "
    "images from the Oxford-IIIT Pet nose dataset. One patch is generated for "
    "each training image each epoch. For a nose label (u, v), the patch center "
    "is formed by adding independent random offsets sampled uniformly from "
    "[-40, 40] pixels to u and v, then rounding to the nearest pixel."
)
add_bullet(
    doc,
    "Patch and target: crop a 128 × 128 region centered at the sampled point; "
    "pixels outside the original image are black. Resize with bilinear "
    "interpolation to 64 × 64. The regression target is ((u−cu)/64, (v−cv)/64).",
)
add_bullet(
    doc,
    "Input normalization: convert to RGB and normalize channels with ImageNet "
    "mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225].",
)
add_bullet(
    doc,
    "Augmented runs: with probability 0.5, horizontally flip the patch and "
    "negate the horizontal target offset; then apply ColorJitter with brightness, "
    "contrast, and saturation strengths of 0.3. No-augmentation runs omit both.",
)
add_bullet(
    doc,
    "Optimization: 20 epochs; mini-batch size 64; shuffled batches; mean squared "
    "error on the two normalized offsets; Adam with learning rate 1×10⁻³ and "
    "weight decay 1×10⁻⁵; cosine annealing with T_max=20.",
)
add_bullet(
    doc,
    "Reproducibility: random seed 42 is applied to Python, NumPy, PyTorch, and "
    "CUDA; DataLoader workers are seeded from their PyTorch worker seeds. "
    "The loader uses two workers and persistent workers when enabled.",
)
doc.add_paragraph(
    "Training uses the full training split and does not use a validation set or "
    "test-set feedback. The loss figures below show the recorded training MSE "
    "over the 20 epochs; the curves decline across training, with the largest "
    "loss reduction occurring early."
)

# Outputs and tables
doc.add_page_break()
doc.add_heading("3. Outputs", level=1)
doc.add_heading("3.1 Accuracy", level=2)
accuracy_rows = [
    ("Baseline (mean)", "0.6", "79.7", "70.0", "848.2", "56.1"),
    ("SnoutNet", "0.8", "57.0", "44.3", "1084.5", "57.4"),
    ("SnoutNet-A", "1.8", "44.8", "37.0", "690.0", "39.6"),
    ("SnoutNet-V", "0.4", "12.9", "8.7", "175.5", "17.3"),
    ("Ensemble (mean)", "0.7", "31.2", "23.5", "597.2", "33.7"),
    ("Ensemble (median)", "0.4", "29.4", "20.2", "676.3", "38.0"),
    ("SnoutNet + corr.", "0.4", "43.8", "19.4", "1062.8", "61.1"),
    ("SnoutNet + corr. (aug)", "0.2", "43.7", "20.9", "1064.6", "61.2"),
    ("SnoutNet + mod. corr.", "0.7", "50.9", "35.9", "1079.4", "59.4"),
    ("SnoutNet + mod. corr. (aug)", "0.8", "50.3", "33.1", "1078.7", "59.6"),
    ("SnoutNet-V + corr.", "0.2", "12.2", "7.7", "202.6", "17.2"),
    ("SnoutNet-V + corr. (aug)", "0.3", "11.8", "7.3", "198.9", "17.0"),
    ("SnoutNet-V + mod. corr.", "0.2", "13.6", "9.5", "181.2", "17.0"),
    ("SnoutNet-V + mod. corr. (aug)", "0.3", "13.0", "8.8", "185.6", "17.2"),
    ("Ensemble (mean) + corr.", "0.3", "20.5", "10.3", "594.8", "34.3"),
    ("Ensemble (mean) + corr. (aug)", "0.5", "20.4", "9.7", "596.8", "34.7"),
    ("Ensemble (mean) + mod. corr.", "0.6", "25.7", "16.5", "596.7", "34.1"),
    ("Ensemble (mean) + mod. corr. (aug)", "0.8", "25.2", "15.5", "597.9", "34.2"),
]
add_table(
    doc,
    ["Model", "Min", "Mean", "Median", "Max", "Std"],
    accuracy_rows,
    [3.15, 0.72, 0.72, 0.8, 0.78, 0.68],
    font_size=7.5,
)
add_caption(
    doc,
    "Table 1. Test-set Euclidean localization error in original-image pixels "
    "(lower is better). Values are reproduced from evaluate.py.",
)
doc.add_paragraph(
    "The maximum-error outliers remain large for the scratch-trained SnoutNet "
    "and for the mean ensemble. Consequently, the mean and standard deviation "
    "are strongly affected by a small number of severe misses; the median is "
    "also reported to show typical-image performance."
)

doc.add_heading("3.2 Computational cost", level=2)
cost_rows = [
    ("Baseline (mean)", "0", "0.0"),
    ("SnoutNet", "5,617,794", "9.3"),
    ("SnoutNet-A", "57,012,034", "15.5"),
    ("SnoutNet-V", "134,268,738", "124.2"),
    ("Ensemble (mean)", "196,898,566", "149.0"),
    ("Ensemble (median)", "196,898,566", "149.0"),
    ("SnoutNet + corr.", "6,384,388", "10.4"),
    ("SnoutNet + corr. (aug)", "6,384,388", "10.5"),
    ("SnoutNet + mod. corr.", "6,736,996", "12.0"),
    ("SnoutNet + mod. corr. (aug)", "6,736,996", "11.9"),
    ("SnoutNet-V + corr.", "135,035,332", "125.2"),
    ("SnoutNet-V + corr. (aug)", "135,035,332", "125.2"),
    ("SnoutNet-V + mod. corr.", "135,387,940", "126.9"),
    ("SnoutNet-V + mod. corr. (aug)", "135,387,940", "126.8"),
    ("Ensemble (mean) + corr.", "197,665,160", "150.1"),
    ("Ensemble (mean) + corr. (aug)", "197,665,160", "150.1"),
    ("Ensemble (mean) + mod. corr.", "198,017,768", "151.6"),
    ("Ensemble (mean) + mod. corr. (aug)", "198,017,768", "151.6"),
]
add_table(
    doc,
    ["Model", "Parameters", "ms / image"],
    cost_rows,
    [4.1, 1.65, 1.1],
    font_size=7.5,
)
add_caption(
    doc,
    "Table 2. Parameter count and per-image inference time reported by "
    "evaluate.py. Latency depends on the execution hardware.",
)

doc.add_heading("3.3 Cumulative error distribution", level=2)
doc.add_paragraph(
    "The CED curve gives the percentage of test predictions whose localization "
    "error is no greater than the horizontal-axis value. Curves that rise more "
    "quickly and lie farther left indicate better localization. The supplied "
    "plot uses a 0–200 pixel horizontal range; the accuracy table above retains "
    "the full error range, including larger outliers."
)
add_figure(
    doc,
    "ced.png",
    "Figure 1. Cumulative error distribution for the in-lab models, baselines, "
    "ensembles, and all four correction configurations.",
)

doc.add_heading("3.4 Best and worst SnoutNet predictions", level=2)
doc.add_paragraph(
    "Each panel shows the four best predictions (top row) and four worst "
    "predictions (bottom row). The red cross is the prediction and the green "
    "circle marks the ground-truth nose location."
)
add_figure(
    doc,
    "best_worst_snoutnet.png",
    "Figure 2. SnoutNet predictions before correction.",
)
add_figure(
    doc,
    "best_worst_snoutnet_correction_noaug.png",
    "Figure 3. SnoutNet predictions corrected with CorrectionNet trained without augmentation.",
)
add_figure(
    doc,
    "best_worst_snoutnet_correction_aug.png",
    "Figure 4. SnoutNet predictions corrected with augmented CorrectionNet.",
)

doc.add_heading("3.5 Training loss curves", level=2)
doc.add_paragraph(
    "The following plots show training MSE on normalized offsets for all four "
    "training configurations."
)
add_figure(
    doc,
    "loss_correction_noaug.png",
    "Figure 5. CorrectionNet without augmentation.",
)
add_figure(
    doc,
    "loss_correction_aug.png",
    "Figure 6. CorrectionNet with horizontal-flip and colour-jitter augmentation.",
)
add_figure(
    doc,
    "loss_correction_mod_noaug.png",
    "Figure 7. CorrectionNetModified without augmentation.",
)
add_figure(
    doc,
    "loss_correction_mod_aug.png",
    "Figure 8. CorrectionNetModified with horizontal-flip and colour-jitter augmentation.",
)

# Analysis
doc.add_page_break()
doc.add_heading("4. Analysis", level=1)
doc.add_heading("4.1 Failure cases", level=2)
doc.add_paragraph(
    "The hardest cases are images where the nose is small, partly occluded, "
    "poorly contrasted, off-center, or presented at an unusual angle or scale. "
    "The worst SnoutNet example is Egyptian_Mau_165.jpg (1,084.5 px error); its "
    "predicted point is far from the face. The worst corrected SnoutNet example "
    "is still Egyptian_Mau_165.jpg (1,064.6 px with augmented correction). This "
    "illustrates that a local second stage cannot recover a nose that is absent "
    "from its input patch. Other severe misses include images with unusual poses "
    "or distracting background content. The model learns a typical face/nose "
    "appearance, but does not have a mechanism to reject every out-of-distribution "
    "or ambiguous image."
)

doc.add_heading("4.2 Effect of pretraining", level=2)
doc.add_paragraph(
    "Pretraining provides reusable edge, texture, shape, and object-part "
    "representations learned from ImageNet, so the model need not learn all "
    "visual features from the comparatively small task-specific dataset. This "
    "is consistent with the results: SnoutNet-V has a 12.9 px mean error, versus "
    "57.0 px for SnoutNet trained from scratch. SnoutNet-A also improves on the "
    "scratch model (44.8 px mean). SnoutNet-V is the most accurate in-lab model, "
    "though its 124.2 ms/image cost is much higher than SnoutNet's 9.3 ms/image."
)

doc.add_heading("4.3 Ensembles", level=2)
doc.add_paragraph(
    "Averaging predictions improves on SnoutNet and SnoutNet-A alone: the mean "
    "ensemble has 31.2 px mean error, and the median ensemble 29.4 px. However, "
    "the ensemble does not outperform SnoutNet-V (12.9 px). Combining models "
    "can dilute the strongest member when the other predictions are less "
    "accurate, and errors can remain correlated because every model localizes "
    "the same feature from the same image. An ensemble is most likely to beat "
    "its best member when the members have complementary, weakly correlated "
    "errors and their outputs are comparably calibrated."
)

doc.add_heading("4.4 Effect of the correction model", level=2)
doc.add_paragraph(
    "The standard correction model reduces the mean error for all three "
    "correctable inputs. For SnoutNet, the augmented version changes mean error "
    "from 57.0 to 43.7 px; for SnoutNet-V, from 12.9 to 11.8 px; and for the "
    "mean ensemble, from 31.2 to 20.4 px. Median error also falls substantially "
    "for the VGG and ensemble inputs (8.7→7.3 px and 23.5→9.7 px, respectively). "
    "The relative gain is largest for the ensemble's median, while the VGG "
    "baseline is already close to the nose and has less room to improve."
)
doc.add_paragraph(
    "Correction is not guaranteed to fix a bad first-stage prediction. Training "
    "offsets are sampled within ±40 pixels per coordinate, and the 128 × 128 "
    "crop spans only ±64 pixels from its center. When the initial prediction is "
    "far outside this neighborhood, the nose may not be visible in the patch or "
    "the required offset may be outside the training distribution. The remaining "
    "large maximum errors in Table 1 show that this local refinement does not "
    "remove catastrophic first-stage failures."
)

doc.add_heading("4.5 Effect of augmentation", level=2)
doc.add_paragraph(
    "Horizontal flipping with a corresponding sign change for the horizontal "
    "target preserves the label while exposing the model to both left-right "
    "orientations. Colour jitter encourages robustness to illumination and "
    "colour variation. For the standard model, augmentation has a small positive "
    "effect on the mean for SnoutNet (43.8→43.7 px), SnoutNet-V (12.2→11.8 px), "
    "and the ensemble (20.5→20.4 px). Median error improves for VGG and the "
    "ensemble, but not for corrected SnoutNet (19.4→20.9 px). Thus, augmentation "
    "is mildly beneficial overall, rather than a uniform improvement on every "
    "summary statistic."
)

doc.add_heading("4.6 Modified architecture", level=2)
doc.add_paragraph(
    "The modified model uses two-convolution residual blocks, channel-projection "
    "skip connections where necessary, and dropout before the output layer. It "
    "increases the correction model from 766,594 to 1,119,202 parameters. Under "
    "the same training recipe, it performs worse than the standard correction "
    "model for each first-stage input: with augmentation, mean errors are 50.3 "
    "versus 43.7 px for SnoutNet, 13.0 versus 11.8 px for SnoutNet-V, and 25.2 "
    "versus 20.4 px for the mean ensemble. The added capacity and residual "
    "structure therefore did not translate into better test localization in "
    "this experiment. A plausible explanation is that the local offset task "
    "and available training set do not benefit from the extra capacity under "
    "the unchanged hyperparameters; without a validation set or controlled "
    "hyperparameter search, this should be treated as an observed outcome, not "
    "a definitive explanation."
)

doc.add_heading("Conclusion", level=1)
doc.add_paragraph(
    "A patch-based correction stage improves typical test localization, most "
    "notably for the mean ensemble, while preserving the original model's "
    "catastrophic failures when the nose is outside the local crop. ImageNet "
    "pretraining remains the strongest contributor to accuracy: SnoutNet-V "
    "starts from the best in-lab performance and is further refined by the "
    "standard correction model. The augmentation gains are modest, and the "
    "larger residual correction architecture did not improve test results with "
    "the training setup used."
)

doc.save(OUTPUT)
print(OUTPUT)

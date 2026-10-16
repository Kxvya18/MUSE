// MUSE story deck: problem -> terms -> method -> reasoning -> code -> results -> next steps.
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const path = require("path");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const FIG = path.join(__dirname, "..", "figures");
const THEME = {
  name: "MUSE",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1F3B", lt1: "FFFFFF", dk2: "3A3F5C", lt2: "EEF2F7",
    accent1: "1F7A8C", accent2: "E07A5F", accent3: "F2CC8F", accent4: "81B29A",
    accent5: "3D405B", accent6: "9C6ADE", hlink: "1F7A8C", folHlink: "9C6ADE",
  },
};
const HEX = { navy: "1B1F3B", teal: "1F7A8C", coral: "E07A5F", sand: "F2CC8F", sage: "81B29A", slate: "3D405B", mist: "EEF2F7", violet: "9C6ADE" };

async function icon(Icon, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Icon, { color: "#" + color, size: String(size) }));
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
  pres.title = "MUSE: the story";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;

  pres.defineSlideMaster({
    title: "TITLE", background: { color: C.text1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 11.7, h: 1.6, fontSize: 44, bold: true, color: C.background1, fontFace: THEME.headFontFace }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.9, w: 11.7, h: 1.4, fontSize: 20, color: C.accent3 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "SECTION", background: { color: C.accent1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.6, w: 11.7, h: 1.3, fontSize: 40, bold: true, color: C.background1, fontFace: THEME.headFontFace }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.9, w: 11.7, h: 1.0, fontSize: 20, color: C.background1 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT", background: { color: C.background1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.8, fontSize: 32, bold: true, color: C.text1, fontFace: THEME.headFontFace }, text: "" } },
    ],
    slideNumber: { x: 12.4, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.text2 },
  });

  let section = "";
  const sec = (t) => { section = t; pres.addSection({ title: t }); };
  const content = (title, notes) => {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
    s.addText(title, { placeholder: "title" });
    if (notes) s.addNotes(notes);
    return s;
  };
  const divider = (title, sub, notes) => {
    const s = pres.addSlide({ masterName: "SECTION", sectionTitle: section });
    s.addText(title, { placeholder: "title" });
    s.addText(sub, { placeholder: "body" });
    if (notes) s.addNotes(notes);
    return s;
  };
  // card: tinted rounded box with icon circle, bold header and body text
  const card = async (s, x, y, w, h, Icon, colour, head, body, name, bodySize = 14) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.12, objectName: name + "-card" });
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.25, w: 0.62, h: 0.62, fill: { color: colour }, line: { color: colour }, objectName: name + "-dot" });
    s.addImage({ data: await icon(Icon, "FFFFFF"), x: x + 0.39, y: y + 0.39, w: 0.34, h: 0.34, objectName: name + "-icon" });
    s.addText(head, { x: x + 1.0, y: y + 0.22, w: w - 1.15, h: 0.65, fontSize: 18, bold: true, color: C.text1, valign: "middle", margin: 0, isTextBox: true, objectName: name + "-head" });
    s.addText(body, { x: x + 0.25, y: y + 1.0, w: w - 0.5, h: h - 1.15, fontSize: bodySize, color: C.text2, valign: "top", margin: 0, isTextBox: true, objectName: name + "-body" });
  };
  const bullets = (items, size = 16) => items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8, fontSize: size } }));
  const term = (s, x, y, w, h, word, meaning, colour, name) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.1, objectName: name + "-box" });
    s.addText([{ text: word + "\n", options: { bold: true, fontSize: 17, color: colour } }, { text: meaning, options: { fontSize: 14, color: C.text2 } }],
      { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: h - 0.24, valign: "top", margin: 0, isTextBox: true, objectName: name + "-text" });
  };

  // ---------------------------------------------------------------- 1. opening
  sec("Opening");
  let s = pres.addSlide({ masterName: "TITLE", sectionTitle: section });
  s.addText("MUSE: trustworthy AI when medical scans are missing", { placeholder: "title" });
  s.addText("Modality-Universal Selective Evidence Fusion — the full story: problem, ideas, code, results", { placeholder: "body" });
  s.addNotes("This deck tells the MUSE story end to end for a mixed audience: why missing scans matter, the plain meaning of every technical term, how MUSE works, why each design decision was made, how the code is built, and what the experiments show — honestly, including where MUSE does not win.");

  s = content("The story in one slide", "Read left to right: a real clinical problem, one idea, a carefully engineered system, and measured results.");
  const steps = [
    [fa.FaHospitalUser, HEX.coral, "1 · Problem", "Brain-tumour AI needs 4 MRI scans, but patients often miss some. Current AI fakes them or breaks."],
    [fa.FaLightbulb, HEX.sand, "2 · Idea", "Treat each scan as an expert giving evidence. A missing scan simply says nothing — no faking."],
    [fa.FaCode, HEX.teal, "3 · Build", "One 3D network per scan, evidence fusion, uncertainty and explanations, tested on 98 real patients."],
    [fa.FaChartLine, HEX.sage, "4 · Result", "Matches the best published-style methods on accuracy, and adds trustworthy uncertainty and explanations."],
  ];
  for (let i = 0; i < 4; i++) await card(s, 0.6 + i * 3.08, 1.6, 2.88, 4.6, steps[i][0], steps[i][1], steps[i][2], steps[i][3], "story" + i, 16);

  // ---------------------------------------------------------------- 2. problem
  sec("The problem");
  divider("Chapter 1 · The problem", "Why missing scans break today's medical AI", "Start with the patient.");
  s = content("A brain-tumour patient needs four kinds of MRI scan", "Each MRI 'sequence' is the same brain photographed with a different physical setting, so each highlights different tissue. Radiologists combine them.");
  const seqs = [
    [fa.FaWater, HEX.teal, "FLAIR", "Highlights swelling (edema) around the tumour."],
    [fa.FaBrain, HEX.slate, "T1", "Shows normal anatomy clearly."],
    [fa.FaSyringe, HEX.coral, "T1gd", "T1 after contrast dye: lights up the active, growing tumour."],
    [fa.FaTint, HEX.violet, "T2", "Fluid-sensitive; shows tumour and swelling together."],
  ];
  for (let i = 0; i < 4; i++) await card(s, 0.6 + i * 3.08, 1.6, 2.88, 3.2, seqs[i][0], seqs[i][1], seqs[i][2], seqs[i][3], "seq" + i, 16);
  s.addText("The AI's job: outline the tumour and its parts on every 3D pixel (voxel) of the brain.", { x: 0.6, y: 5.2, w: 12.1, h: 0.7, fontSize: 18, italic: true, color: C.accent1, margin: 0, isTextBox: true, objectName: "seq-job" });

  s = content("In real hospitals, scans are often missing", "Emergencies, contrast allergies, scanner availability and different hospital protocols mean many patients lack one or more sequences. With 4 sequences there are 15 possible combinations of what is available.");
  s.addText([{ text: "15", options: { fontSize: 72, bold: true, color: C.accent2 } }, { text: "\npossible combinations of available scans", options: { fontSize: 16, color: C.text2 } }],
    { x: 0.6, y: 1.6, w: 3.6, h: 2.6, margin: 0, isTextBox: true, objectName: "combos" });
  s.addText("How today's AI copes — and why it fails", { x: 4.6, y: 1.5, w: 8.1, h: 0.5, fontSize: 20, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: "fail-head" });
  s.addText(bullets([
    "Fake the missing scan with a generative AI — the model then trusts invented data",
    "Fill it with zeros — the network never saw that in training and fails silently",
    "Train 15 separate models — impractical to build, validate and deploy",
    "Drop the patient — the patients who most need help are excluded",
  ], 17), { x: 4.6, y: 2.1, w: 8.1, h: 3.4, color: C.text2, margin: 0, isTextBox: true, objectName: "fail-list" });

  // ---------------------------------------------------------------- 3. terms
  sec("Key terms");
  divider("Chapter 2 · The vocabulary", "Every technical word, in plain language", "Four glossary slides: images, networks, evidence, and how we judge results.");
  s = content("Words about images and results");
  const t1 = [
    ["Segmentation", "Colouring every voxel by what it is (healthy, swelling, tumour core, active tumour)."],
    ["Voxel", "A 3D pixel — here 1 mm × 1 mm × 1 mm of brain."],
    ["WT / TC / ET", "Whole tumour, tumour core, enhancing (active) tumour — the three regions doctors care about."],
    ["Dice score", "Overlap between AI outline and expert outline: 0 = none, 1 = perfect."],
    ["HD95", "How far (mm) the AI's border strays from the true border; lower is better."],
    ["Missing modality", "A scan type that was not acquired for this patient."],
  ];
  t1.forEach((t, i) => term(s, 0.6 + (i % 3) * 4.1, 1.5 + Math.floor(i / 3) * 2.75, 3.9, 2.5, t[0], t[1], C.accent1, "t1-" + i));

  s = content("Words about neural networks");
  const t2 = [
    ["Neural network", "A program that learns patterns from examples instead of hand-written rules."],
    ["3D U-Net", "The standard medical-image network: shrinks the scan to see context, then expands back to full detail."],
    ["Encoder / decoder", "The shrinking half (understands) and the expanding half (draws the outline)."],
    ["Training / validation / test", "Learn on one set, choose settings on a second, report on a third never seen before."],
    ["Seed", "The random starting point of training; repeating with several seeds shows results are not luck."],
    ["Graph neural network", "A network for data connected like a web (papers citing papers); used for MUSE's second test bed."],
  ];
  t2.forEach((t, i) => term(s, 0.6 + (i % 3) * 4.1, 1.5 + Math.floor(i / 3) * 2.75, 3.9, 2.5, t[0], t[1], C.accent5, "t2-" + i));

  s = content("Words about evidence and uncertainty", "These are the heart of MUSE.");
  const t3 = [
    ["Evidence (α)", "How strongly a scan supports each class. α = 1 means 'no evidence at all'."],
    ["Dirichlet distribution", "The maths that turns evidence counts into probabilities plus uncertainty."],
    ["Uncertainty u = C / S", "Number of classes ÷ total evidence: little evidence → high uncertainty."],
    ["Dempster–Shafer theory", "A theory of evidence that can say 'I don't know' instead of forcing a guess."],
    ["Vacuous mass", "The formal 'I know nothing' opinion — exactly what a missing scan contributes."],
    ["Cumulative fusion", "Combine opinions by adding their evidence (from subjective logic)."],
  ];
  t3.forEach((t, i) => term(s, 0.6 + (i % 3) * 4.1, 1.5 + Math.floor(i / 3) * 2.75, 3.9, 2.5, t[0], t[1], C.accent2, "t3-" + i));

  s = content("Words about judging results fairly");
  const t4 = [
    ["Baseline", "An existing method we compare against (HeMIS, RFNet, mmFormer, …)."],
    ["Ablation", "Remove one part of MUSE and re-test, to prove that part actually helps."],
    ["Calibration (ECE)", "Does 90% confidence mean right 90% of the time? Lower error = more honest."],
    ["AUROC (error detection)", "How well the uncertainty points at the AI's own mistakes; 1 = perfectly."],
    ["p-value (Wilcoxon test)", "Chance the difference is luck; below 0.05 counts as significant."],
    ["Pull request (PR)", "A reviewed, recorded change to the code on GitHub — our audit trail."],
  ];
  t4.forEach((t, i) => term(s, 0.6 + (i % 3) * 4.1, 1.5 + Math.floor(i / 3) * 2.75, 3.9, 2.5, t[0], t[1], C.accent4, "t4-" + i));

  // ---------------------------------------------------------------- 4. method
  sec("How MUSE works");
  divider("Chapter 3 · How MUSE works", "A panel of experts that never fakes evidence");
  s = content("MUSE architecture: one forward pass, three outputs", "Each available scan goes through its own 3D U-Net and evidential head. Missing scans skip the network and contribute alpha = 1 (no evidence). A gate learns how much to trust each scan for each class at each voxel. Evidence is summed. Outputs: segmentation, uncertainty map, per-scan contributions.");
  s.addImage({ path: path.join(FIG, "muse_architecture.png"), x: 0.6, y: 1.45, w: 8.6, h: 4.1, objectName: "arch" });
  s.addText(bullets([
    "Own network per scan",
    "Missing scan = vacuous mass",
    "Gate = trust per class & voxel",
    "Fusion = add the evidence",
    "Outputs: outline, uncertainty, who-said-what",
  ], 16), { x: 9.5, y: 1.6, w: 3.3, h: 4.2, color: C.text2, margin: 0, isTextBox: true, objectName: "arch-steps" });

  s = content("The same idea as a panel of four experts");
  const exp = [
    [fa.FaUserMd, HEX.teal, "Each expert reads one scan", "FLAIR expert, T1 expert, … each marks where it thinks the tumour is."],
    [fa.FaVolumeMute, HEX.slate, "Missing scan → expert stays silent", "No guessing, no invented data. Silence adds exactly nothing."],
    [fa.FaBalanceScale, HEX.coral, "Votes are weighted by trust", "The contrast-scan expert counts more for active tumour — MUSE learns this."],
    [fa.FaPlusCircle, HEX.sage, "Weighted votes are added up", "Agreement → confident. Little evidence or conflict → uncertain, and MUSE says so."],
  ];
  for (let i = 0; i < 4; i++) await card(s, 0.6 + (i % 2) * 6.15, 1.5 + Math.floor(i / 2) * 2.75, 5.95, 2.5, exp[i][0], exp[i][1], exp[i][2], exp[i][3], "exp" + i, 16);

  s = content("Three guarantees MUSE can prove", "Because fusion is a sum of non-negative evidence: order does not matter, adding zero changes nothing, and adding evidence can only lower uncertainty. We also checked all three on real model outputs: zero violations.");
  const g = [
    ["Order never matters", "Fusing FLAIR then T2, or T2 then FLAIR, gives the identical answer (checked: difference 0)."],
    ["Missing scans are harmless", "A missing scan adds exactly zero — one model serves all 15 combinations."],
    ["More scans never raise doubt", "Adding a scan can only lower uncertainty (checked on real patients: 0 violations)."],
  ];
  g.forEach((t, i) => term(s, 0.6 + i * 4.1, 1.6, 3.9, 3.0, t[0], t[1], C.accent1, "g" + i));
  s.addText("Honest note: these follow from known 'cumulative fusion' maths; MUSE's novelty is the complete trustworthy system built on them.", { x: 0.6, y: 5.0, w: 12.1, h: 0.8, fontSize: 15, italic: true, color: C.text2, margin: 0, isTextBox: true, objectName: "g-note" });

  // ---------------------------------------------------------------- 5. reasoning
  sec("Reasoning journey");
  divider("Chapter 4 · The reasoning journey", "What we changed, and why — decided on validation data only");
  s = content("First, we corrected the original plan", "The design documents contained claims a reviewer would reject. We fixed them rather than copy them.");
  await card(s, 0.6, 1.5, 3.95, 4.4, fa.FaTimesCircle, HEX.coral, "False claim removed", "'Dempster's rule is order-dependent' — it is not. We proved and measured it: difference 3×10⁻¹⁶.", "fix1", 15);
  await card(s, 4.69, 1.5, 3.95, 4.4, fa.FaExchangeAlt, HEX.sand, "Wrong sign fixed", "The 'balance' term pushed scans apart instead of balancing them. Fixed; the original is kept as an ablation.", "fix2", 15);
  await card(s, 8.78, 1.5, 3.95, 4.4, fa.FaBalanceScaleLeft, HEX.teal, "Honest novelty", "Evidence-sum fusion already exists; MUSE's contribution is the system: gate, vacuous mass, uncertainty, explanations.", "fix3", 15);

  s = content("Each design step was earned on validation data", "Graph benchmark (Cora) validation accuracy, then the move to real 3D MRI.");
  s.addChart(pres.charts.BAR, [{ name: "Validation accuracy", labels: ["Original plan", "+ neighbour evidence", "+ per-scan training", "+ tuning"], values: [0.696, 0.718, 0.770, 0.791] }], {
    x: 0.6, y: 1.5, w: 7.2, h: 4.6, barDir: "col", chartColors: [HEX.teal], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelFormatCode: "0.000",
    valAxisMinVal: 0.6, valAxisMaxVal: 0.85, valGridLine: { color: "D9DEE7", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", catAxisLabelFontSize: 12, valAxisLabelFontSize: 12,
    showTitle: true, title: "Cora validation accuracy", titleFontSize: 14, titleFontFace: "+mn-lt", objectName: "journey-chart",
  });
  s.addText(bullets([
    "Bug found: MUSE could not learn on Amazon-Photo → added normalisation",
    "MRI as coarse graphs capped accuracy → rebuilt as full 3D U-Nets",
    "Same fixes given to baselines that broke (ECD, EGNN) — fairness",
  ], 15), { x: 8.1, y: 1.7, w: 4.6, h: 4.2, color: C.text2, margin: 0, isTextBox: true, objectName: "journey-notes" });

  // ---------------------------------------------------------------- 6. code
  sec("How the code is built");
  divider("Chapter 5 · How the code is built", "From raw scans to tables and figures, reproducibly");
  s = content("The pipeline, end to end", "Raw MRI download, preprocessing (crop to brain, normalise each scan), random 3D crops with randomly hidden scans during training, model selection on validation patients, testing on 98 unseen patients across all 15 combinations, then tables and figures straight from the saved results.");
  const pipe = [["Download", "Public MSD brain-tumour MRI, 484 patients"], ["Preprocess", "Crop to brain, normalise each scan"], ["Train", "3D crops, random scans hidden, 12,000 steps"], ["Validate", "Pick best checkpoint on unseen validation patients"], ["Test", "98 patients × 15 scan combinations"], ["Report", "Tables, significance tests, real-scan figures"]];
  pipe.forEach((p, i) => {
    const x = 0.6 + i * 2.07;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.0, w: 1.85, h: 2.9, fill: { color: i % 2 ? C.accent5 : C.accent1 }, line: { color: i % 2 ? C.accent5 : C.accent1 }, rectRadius: 0.12, objectName: "pipe-" + i });
    s.addText([{ text: `${i + 1}\n`, options: { fontSize: 26, bold: true, color: C.background1 } }, { text: p[0] + "\n", options: { fontSize: 16, bold: true, color: C.background1 } }, { text: p[1], options: { fontSize: 13, color: C.background1 } }],
      { x: x + 0.12, y: 2.15, w: 1.61, h: 2.6, valign: "top", margin: 0, isTextBox: true, objectName: "pipe-text-" + i });
  });
  s.addText("Every number in the papers is produced by this code from saved result files — never typed by hand.", { x: 0.6, y: 5.4, w: 12.1, h: 0.6, fontSize: 16, italic: true, color: C.accent1, margin: 0, isTextBox: true, objectName: "pipe-note" });

  s = content("How the code is organised", "Small, readable modules. Each file has one job.");
  const files = [
    ["muse3d/models.py", "MUSE and every MRI baseline on one shared 3D U-Net backbone"],
    ["muse3d/losses.py", "Evidence loss + Dice loss; each model's own training objective"],
    ["run_mri3d.py", "Preprocess → train (resumable) → test all 15 combinations"],
    ["muse/ & run_graph.py", "Graph version of MUSE and the citation / product benchmarks"],
    ["aggregate*.py, figures/", "Tables with significance tests, and every figure"],
    ["tests/", "14 automatic checks, e.g. a missing scan can never change the output"],
  ];
  files.forEach((f, i) => {
    const y = 1.5 + i * 0.85;
    s.addText(f[0], { x: 0.6, y, w: 3.8, h: 0.7, fontSize: 16, bold: true, fontFace: "Courier New", color: C.accent1, valign: "middle", margin: 0, isTextBox: true, objectName: "file-" + i });
    s.addText(f[1], { x: 4.6, y, w: 8.1, h: 0.7, fontSize: 16, color: C.text2, valign: "middle", margin: 0, isTextBox: true, objectName: "file-desc-" + i });
  });

  s = content("How we kept it rigorous");
  const rig = [
    [fa.FaVial, HEX.teal, "Tested code", "14 automatic tests: properties hold, missing scans have zero effect, fast paths equal slow paths."],
    [fa.FaEyeSlash, HEX.coral, "No peeking at test data", "All choices on validation data; test results computed once at the end."],
    [fa.FaCodeBranch, HEX.slate, "Every change reviewed", "Feature branch → pull request → review → merge: a complete audit trail on GitHub."],
    [fa.FaServer, HEX.sage, "Compute where it fit", "Mac for testing, free Colab and Kaggle GPUs for training, resumable from checkpoints."],
  ];
  for (let i = 0; i < 4; i++) await card(s, 0.6 + (i % 2) * 6.15, 1.5 + Math.floor(i / 2) * 2.75, 5.95, 2.5, rig[i][0], rig[i][1], rig[i][2], rig[i][3], "rig" + i, 16);

  // ---------------------------------------------------------------- 7. results
  sec("Results");
  divider("Chapter 6 · What we found", "98 unseen patients, 15 scan combinations, 7 methods");
  const HL = path.join(FIG, "highlights"), RL = path.join(FIG, "realistic"), GR = path.join(FIG, "results");
  const img = (sl, file, x, y, w, h, name) => sl.addImage({ path: file, x, y, w, h, sizing: { type: "contain", w, h }, objectName: name });

  s = content("Brain-tumour MRI: overall accuracy", "Mean Dice over whole tumour, tumour core and enhancing tumour, averaged over all 15 combinations. MUSE vs mmFormer-style and RFNet-style: no significant difference (a tie). MUSE vs HeMIS, LateMean, UNet-MD: significantly better.");
  s.addChart(pres.charts.BAR, [{ name: "Mean Dice", labels: ["mmFormer-style", "MUSE", "RFNet-style", "HeMIS", "TMC", "LateMean", "UNet-MD"], values: [0.614, 0.612, 0.606, 0.603, 0.601, 0.594, 0.507] }], {
    x: 0.6, y: 1.45, w: 7.6, h: 4.9, barDir: "bar", chartColors: [HEX.teal], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelFormatCode: "0.000",
    valAxisMinVal: 0.4, valAxisMaxVal: 0.65, valGridLine: { color: "D9DEE7", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, catAxisOrientation: "maxMin",
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", catAxisLabelFontSize: 13, valAxisLabelFontSize: 12,
    showTitle: true, title: "Mean Dice (higher is better)", titleFontSize: 14, titleFontFace: "+mn-lt", objectName: "mri-chart",
  });
  s.addText([{ text: "Tie at the top\n", options: { fontSize: 20, bold: true, color: C.text1 } },
    { text: "MUSE matches the strongest specialised methods and significantly beats HeMIS, LateMean and the standard U-Net.", options: { fontSize: 16, color: C.text2 } }],
    { x: 8.6, y: 1.8, w: 4.1, h: 3.6, valign: "top", margin: 0, isTextBox: true, objectName: "mri-msg" });

  s = content("Each method's strengths at a glance", "Each axis is scaled from the worst to the best of these four methods, so differences look larger than they are; the actual numbers are on the previous and next slides. MUSE leads on whole tumour, tumour core, error detection and single-sequence robustness; mmFormer-style on enhancing tumour and calibration.");
  img(s, path.join(HL, "radar_mri.png"), 0.6, 1.35, 6.4, 5.7, "radar");
  s.addText(bullets(["MUSE leads: whole tumour, tumour core, error detection, single-scan robustness", "mmFormer-style leads: enhancing tumour, calibration", "Axes scaled worst → best of these four, so gaps look larger than they are"], 16),
    { x: 7.4, y: 1.8, w: 5.3, h: 4.4, color: C.text2, margin: 0, isTextBox: true, objectName: "radar-notes" });

  s = content("Every scan combination, every method", "Mean Dice for each of the 15 combinations of available scans (red box = best in that column). The top methods are usually within 0.01 of each other; MUSE is best with all scans but T2 and with FLAIR alone.");
  img(s, path.join(HL, "heatmap_patterns.png"), 0.4, 1.35, 12.5, 5.3, "heatmap");

  s = content("MUSE helps most where segmentation is hardest", "Left: Dice as scans go missing; all strong methods degrade similarly. Right: each dot is one patient. mmFormer-style wins more patients overall, mostly easy ones; MUSE wins most of the hard, low-Dice patients.");
  img(s, path.join(HL, "robustness_curve.png"), 0.4, 1.4, 6.4, 4.6, "robust");
  img(s, path.join(HL, "paired_scatter.png"), 7.0, 1.4, 5.8, 4.6, "paired");
  s.addText("On hard patients (lower-left), MUSE is mostly above the line.", { x: 0.6, y: 6.25, w: 12.1, h: 0.5, fontSize: 16, italic: true, color: C.accent1, margin: 0, isTextBox: true, objectName: "paired-msg" });

  s = content("MUSE knows when it is wrong — and shows why", "Left: how well each method's uncertainty flags its own mistakes, per patient — MUSE is clearly best. Right: the share of evidence MUSE drew from each scan on tumour voxels, matching how radiologists read these scans.");
  img(s, path.join(HL, "error_detection.png"), 0.4, 1.4, 7.6, 4.5, "errdet");
  img(s, path.join(HL, "contributions_donut.png"), 8.3, 1.4, 4.6, 4.6, "donut");

  s = content("A real unseen patient, scans removed one by one", "Zoomed real MRI with outlines: green = whole tumour, red = tumour core, gold = enhancing tumour. Columns: expert, MUSE, HeMIS, MUSE's uncertainty (expert outline dashed). As scans disappear, MUSE stays closer to the expert and its uncertainty lights up exactly where it is unsure.");
  img(s, path.join(RL, "BRATS_448_2d_contours.png"), 1.9, 1.3, 9.5, 5.9, "contours2d");

  s = content("The tumour in 3D: expert vs MUSE", "Real 3D surfaces inside a translucent brain: swelling (green), tumour core (red) and enhancing tumour (gold). From left: expert, MUSE with all four scans, without T1gd, with FLAIR only.");
  img(s, path.join(RL, "BRATS_448_3d_regions.png"), 0.4, 1.6, 12.5, 3.65, "regions3d");
  s.addText("With all four scans MUSE reproduces the expert's tumour shape; with fewer scans it degrades gracefully instead of breaking.", { x: 0.6, y: 5.6, w: 12.1, h: 0.6, fontSize: 16, italic: true, color: C.accent1, margin: 0, isTextBox: true, objectName: "regions-msg" });

  s = content("MUSE's doubt, painted on the tumour", "MUSE's uncertainty on the 3D tumour surface. Dark = confident, bright = unsure. Removing scans makes the doubt grow exactly at the tumour border, where mistakes happen.");
  img(s, path.join(RL, "BRATS_448_3d_uncertainty.png"), 0.4, 1.6, 12.5, 4.3, "unc3d");

  s = content("The same idea works beyond MRI: graph data", "Citation and product graphs with four feature groups as 'modalities', randomly missing per node. Test accuracy over 0–70% missing, 5 seeds; MUSE degrades least as features go missing.");
  img(s, path.join(GR, "fig_missing_rate.png"), 0.4, 1.35, 8.2, 5.3, "graph-curves");
  const stats = [["CiteSeer", "69.5%", "best of 9", HEX.teal], ["Amazon-Photo", "89.1%", "best of 9", HEX.teal], ["Cora", "79.0%", "2nd (APPNP 80.7%)", HEX.coral]];
  stats.forEach((t, i) => {
    s.addText([{ text: t[0] + "  ", options: { fontSize: 15, bold: true, color: C.text1 } }, { text: t[1] + "\n", options: { fontSize: 30, bold: true, color: t[3] } }, { text: t[2], options: { fontSize: 13, color: C.text2 } }],
      { x: 8.9, y: 1.6 + i * 1.65, w: 3.9, h: 1.45, valign: "middle", margin: 0, isTextBox: true, objectName: "gstat-" + i });
  });

  s = content("Ablations prove each part earns its place", "Remove a part, retrain, retest on the same 98 patients. p-values are paired Wilcoxon tests against full MUSE.");
  s.addChart(pres.charts.BAR, [{ name: "Mean Dice", labels: ["MUSE (full)", "Sum → Dempster's rule", "No reliability gate"], values: [0.612, 0.604, 0.586] }], {
    x: 0.6, y: 1.5, w: 7.2, h: 4.6, barDir: "col", chartColors: [HEX.teal], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelFormatCode: "0.000",
    valAxisMinVal: 0.55, valAxisMaxVal: 0.63, valGridLine: { color: "D9DEE7", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", catAxisLabelFontSize: 13, valAxisLabelFontSize: 12,
    showTitle: true, title: "MRI mean Dice", titleFontSize: 14, titleFontFace: "+mn-lt", objectName: "abl-chart",
  });
  s.addText(bullets(["Removing the gate costs 2.6 points (p = 6×10⁻¹¹)", "Dempster's rule instead of sum costs 0.8 points (p = 0.009)", "Fusion order: MUSE and Dempster give identical answers in any order; Yager, PCR5 and averaging do not"], 16),
    { x: 8.1, y: 1.8, w: 4.6, h: 4.2, color: C.text2, margin: 0, isTextBox: true, objectName: "abl-list" });

  // ---------------------------------------------------------------- 8. next
  sec("Where we are");
  divider("Chapter 7 · Where we are", "Status, the open question, and the papers");
  s = content("Status and next steps");
  const nx = [
    [fa.FaCheckCircle, HEX.sage, "Done", "All experiments: 7 MRI methods × 98 patients, 9 graph methods × 3 datasets × 5 seeds, ablations, real-scan figures."],
    [fa.FaHourglassHalf, HEX.sand, "Running now", "MUSE2: lets scans share context inside the network, aiming to pull clearly ahead of mmFormer. Kept only if it wins on validation."],
    [fa.FaFileAlt, HEX.teal, "Papers", "IEEE Transactions journal (method + theory + MRI + graphs) and CARES 2027 (MRI + explainability)."],
    [fa.FaQuestionCircle, HEX.coral, "Reviewers may ask", "More training seeds for MRI, and published RFNet/mmFormer numbers on BraTS — we will address both."],
  ];
  for (let i = 0; i < 4; i++) await card(s, 0.6 + (i % 2) * 6.15, 1.5 + Math.floor(i / 2) * 2.75, 5.95, 2.5, nx[i][0], nx[i][1], nx[i][2], nx[i][3], "next" + i, 15);

  s = pres.addSlide({ masterName: "TITLE", sectionTitle: section });
  s.addText("MUSE never fakes a missing scan", { placeholder: "title" });
  s.addText("It says what it knows, how sure it is, and which scan told it — at the accuracy of the best specialised methods.", { placeholder: "body" });

  const out = path.join(__dirname, "MUSE_story.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("written", out);
})();

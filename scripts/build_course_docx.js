// 课设报告 v2.0 Word 构建器（docx-js）
// 输入: 报告/课设报告_v2.0.md（课设八段式结构正文）
//       -> 解析为标题/段落/表格/图片/题注 -> 报告/课设报告_v2.0.docx
// 用法: node scripts/build_course_docx.js
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow,
  TableCell, WidthType, AlignmentType, ImageRun, ShadingType, BorderStyle,
} = require("docx");

const SRC = "/home/ctgu53/work/cv-course-design/报告/课设报告_v2.0.md";
const FIGDIR = "/home/ctgu53/work/cv-course-design/outputs/figures/";
const OUT = "/home/ctgu53/work/cv-course-design/报告/课设报告_v2.0.docx";
const DIMS = {
  "fig1_dataset_v1.0.png": [1841, 1408], "fig2_pipeline_v1.0.png": [1710, 776],
  "fig3_training_v1.0.png": [1827, 699], "fig4_qualitative_v1.0.png": [1752, 866],
  "fig5_results_v1.0.png": [1839, 715], "fig6_sam_v1.0.png": [1942, 725],
  "fig7_modules_v1.0.png": [1708, 891], "fig8_runsteps_v2.0.png": [2128, 1437],
  "fig9_archs_v2.0.png": [1708, 984], "fig10_gitph_v2.0.png": [1708, 984],
  "fig11_env_v2.0.png": [1708, 601],
};
const TARGET_W = 620; // 正文图片宽(px 级换算，A4 版心内)

const cells = fs.readFileSync(SRC, "utf-8").split("\n\n");

// ---------- markdown 子集解析 ----------
function inlineRuns(text, base = {}) {
  // 处理 **bold**、`code`、[文字](链接) 的行内格式
  const runs = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) runs.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) {
      runs.push(new TextRun({ text: tok.slice(2, -2), bold: true, ...base }));
    } else if (tok.startsWith("`")) {
      runs.push(new TextRun({ text: tok.slice(1, -1), font: { ascii: "Consolas", hAnsi: "Consolas" }, ...base }));
    } else {
      const mm = /\[([^\]]+)\]\(([^)]+)\)/.exec(tok);
      runs.push(new TextRun({ text: `${mm[1]}（${mm[2]}）`, ...base }));
    }
    last = m.index + tok.length;
  }
  if (last < text.length) runs.push(new TextRun({ text: text.slice(last), ...base }));
  return runs;
}

function mdTable(rows) {
  // rows: 已经 split 好的 | 单元格矩阵（含表头，分隔行已滤除）
  const nCol = rows[0].length;
  const total = 9026;
  const colW = Math.floor(total / nCol);
  const widths = Array(nCol).fill(colW);
  const trs = rows.map((r, ri) => new TableRow({
    tableHeader: ri === 0,
    children: r.map(cell => new TableCell({
      width: { size: colW, type: WidthType.DXA },
      shading: ri === 0 ? { type: ShadingType.CLEAR, fill: "EFEFEF" } : undefined,
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      children: [new Paragraph({
        alignment: ri === 0 ? AlignmentType.CENTER : AlignmentType.LEFT,
        children: inlineRuns(cell.trim(), { size: 17, bold: ri === 0 ? true : undefined }),
      })],
    })),
  }));
  return new Table({ rows: trs, columnWidths: widths, width: { size: total, type: WidthType.DXA } });
}

function imageBlock(alt, fname) {
  const [w, h] = DIMS[fname];
  const scale = TARGET_W / w;
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
    children: [new ImageRun({
      type: "png",
      data: fs.readFileSync(FIGDIR + fname.split("_v")[0] + "/" + fname),
      transformation: { width: Math.round(TARGET_W), height: Math.round(h * scale) },
    })],
  });
}

const children = [];
let tableBuf = [];

function flushTable() {
  if (tableBuf.length) {
    children.push(mdTable(tableBuf));
    children.push(new Paragraph({ text: "", spacing: { after: 80 } }));
    tableBuf = [];
  }
}

for (const cell of cells) {
  const lines = cell.split("\n");
  for (const line of lines) {
    const t = line.trim();
    // 表格行收集
    if (t.startsWith("|")) {
      const cells_ = t.replace(/^\||\|$/g, "").split("|").map(s => s.trim());
      if (cells_.every(c => /^[-: ]+$/.test(c))) continue; // 分隔行
      tableBuf.push(cells_);
      continue;
    }
    flushTable();
    if (!t) continue;
    if (t === "---") continue;
    if (t.startsWith("### ")) {
      children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 160, after: 80 }, children: [new TextRun({ text: t.slice(4), bold: true, size: 24 })] }));
    } else if (t.startsWith("## ")) {
      children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 240, after: 120 }, children: [new TextRun({ text: t.slice(3), bold: true, size: 30 })] }));
    } else if (t.startsWith("# ")) {
      children.push(new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { before: 60, after: 160 },
        children: [new TextRun({ text: t.slice(2), bold: true, size: 32 })],
      }));
    } else if (t.startsWith("> ")) {
      children.push(new Paragraph({
        indent: { left: 360 }, spacing: { after: 80 },
        border: { left: { style: BorderStyle.SINGLE, size: 12, color: "999999", space: 8 } },
        children: inlineRuns(t.slice(2), { size: 19, color: "444444", italics: true }),
      }));
    } else if (t.startsWith("- ")) {
      children.push(new Paragraph({ bullet: { level: 0 }, children: inlineRuns(t.slice(2)) }));
    } else if (t.startsWith("![")) {
      const m = /!\[([^\]]*)\]\(([^)]+)\)/.exec(t);
      const fname = m[2].split("/").pop();
      children.push(imageBlock(m[1], fname));
    } else if (/^\*\*图 \d/.test(t) || /^\*\*表 \d/.test(t)) {
      children.push(new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { after: 120 },
        children: inlineRuns(t, { size: 18 }),
      }));
    } else {
      children.push(new Paragraph({
        alignment: AlignmentType.JUSTIFIED, spacing: { after: 100, line: 300 },
        children: inlineRuns(t),
      }));
    }
  }
  flushTable();
}

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { size: 21, font: { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "SimSun" } },
      },
    },
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 }, // A4
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log("docx 写入:", OUT, buf.length, "bytes");
});

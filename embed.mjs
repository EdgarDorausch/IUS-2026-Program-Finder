#!/usr/bin/env node
/**
 * Precompute paper embeddings for index.html -> embeddings.json.
 *
 * Runs the SAME model, weights and dtype the browser uses (Xenova/bge-small-en-v1.5, q8 ONNX),
 * so document vectors and query vectors live in exactly the same space. Using a different
 * runtime on either side distorts the tail of the score distribution, which is precisely
 * where the ranking cutoff operates.
 *
 * Vectors are unit-normalized, then quantized to int8 (v * 127) and base64-encoded.
 */
import { pipeline } from "@huggingface/transformers";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = dirname(fileURLToPath(import.meta.url));
const MODEL = "Xenova/bge-small-en-v1.5";
const DIM = 384;
const BATCH = 16;

// Documents are embedded bare; BGE's instruction prefix applies to queries only.
const docText = p =>
  [p.title, p.keywords?.join(", "), p.abstract].filter(Boolean).join("\n");

const data = JSON.parse(readFileSync(join(ROOT, "data.json"), "utf8"));

// Flatten order must match template.html's `papers` array exactly.
const papers = [];
for (const s of data.sessions) for (const p of s.papers) papers.push(p);

const missing = papers.filter(p => !p.abstract).length;
console.log(`${papers.length} papers (${missing} without an abstract), model ${MODEL}`);
if (missing) console.log(`  note: papers without an abstract fall back to title + keywords`);

const extractor = await pipeline("feature-extraction", MODEL, { dtype: "q8" });

const out = new Int8Array(papers.length * DIM);
const t0 = Date.now();
let clipped = 0;

for (let i = 0; i < papers.length; i += BATCH) {
  const batch = papers.slice(i, i + BATCH);
  const res = await extractor(batch.map(docText), { pooling: "mean", normalize: true });
  const flat = res.data;
  for (let b = 0; b < batch.length; b++) {
    for (let j = 0; j < DIM; j++) {
      const q = Math.round(flat[b * DIM + j] * 127);
      if (q > 127 || q < -127) clipped++;
      out[(i + b) * DIM + j] = Math.max(-127, Math.min(127, q));
    }
  }
  const done = Math.min(i + BATCH, papers.length);
  if (done % 160 === 0 || done === papers.length) {
    const rate = done / ((Date.now() - t0) / 1000);
    console.log(`  ${done}/${papers.length}  (${rate.toFixed(1)}/s)`);
  }
}

// Fingerprint ties the vectors to the exact paper set; the client refuses to use them if it drifts.
const fp = papers.reduce((a, p) => (a + p.id) >>> 0, 0);

writeFileSync(join(ROOT, "embeddings.json"), JSON.stringify({
  model: MODEL, d: DIM, n: papers.length, fp,
  b64: Buffer.from(out.buffer).toString("base64"),
}));

console.log(`\nembeddings.json written: ${papers.length}x${DIM} int8, fp=${fp}, ${clipped} clipped`);
console.log(`  ${(out.length / 1048576).toFixed(2)} MB raw -> ${(out.length * 4 / 3 / 1048576).toFixed(2)} MB base64`);

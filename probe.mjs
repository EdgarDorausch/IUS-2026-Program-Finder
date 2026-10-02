#!/usr/bin/env node
/**
 * Sanity-check retrieval quality against the built embeddings, without a browser.
 *
 * The probe queries are deliberately ones that exact substring search FAILS on: if these
 * don't surface the right papers, the model or the embedded text is wrong, and it is much
 * cheaper to discover that here than in the UI.
 *
 * A query whose top hit barely rises above the corpus mean (gap < 0.13) is flagged WEAK --
 * measured over this data, genuine queries land at 0.14-0.25 and off-topic ones at 0.08-0.12.
 *
 * Usage: node probe.mjs ["your own query" ...]
 */
import { pipeline } from "@huggingface/transformers";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = dirname(fileURLToPath(import.meta.url));
const QPREFIX = "Represent this sentence for searching relevant passages: ";

const data = JSON.parse(readFileSync(join(ROOT, "data.json"), "utf8"));
const emb = JSON.parse(readFileSync(join(ROOT, "embeddings.json"), "utf8"));
const papers = [];
for (const s of data.sessions) for (const p of s.papers) papers.push(p);

const fp = papers.reduce((a, p) => (a + p.id) >>> 0, 0);
console.log(`vectors ${emb.n}x${emb.d}  papers ${papers.length}  fp ${emb.fp === fp ? "match" : "MISMATCH"}`);
if (emb.n !== papers.length || emb.fp !== fp) process.exit(1);

const V = new Int8Array(Buffer.from(emb.b64, "base64").buffer.slice(0));

// Dequantized rows should still be unit length.
let worst = 0;
for (let i = 0; i < emb.n; i++) {
  let s = 0;
  for (let j = 0; j < emb.d; j++) { const x = V[i * emb.d + j] / 127; s += x * x; }
  worst = Math.max(worst, Math.abs(Math.sqrt(s) - 1));
}
console.log(`max norm deviation after int8 round-trip: ${worst.toFixed(4)}\n`);

const QUERIES = process.argv.slice(2).length ? process.argv.slice(2) : [
  "heart imaging",                        // should surface cardiac / echocardiography
  "AI",                                   // should surface deep learning / neural networks
  "bubbles for drug delivery",            // should surface microbubbles / therapeutic ultrasound
  "treating the brain without surgery",   // should surface transcranial / neuromodulation / HIFU
  "making images from fewer channels",    // should surface sparse arrays / compressed sensing
];

const extractor = await pipeline("feature-extraction", "Xenova/bge-small-en-v1.5", { dtype: "q8" });

for (const q of QUERIES) {
  const o = await extractor(QPREFIX + q, { pooling: "mean", normalize: true });
  const qv = o.data;
  const rows = papers.map((p, i) => {
    let dot = 0;
    for (let j = 0; j < emb.d; j++) dot += V[i * emb.d + j] * qv[j];
    return { p, s: dot / 127 };
  });
  // Mirrors rescore() in template.html: cut relative to this query's own spread.
  const mean = rows.reduce((a, r) => a + r.s, 0) / rows.length;
  rows.sort((a, b) => b.s - a.s);
  const gap = Math.max(1e-6, rows[0].s - mean);
  const cut = Math.min(rows.length, Math.max(10, Math.min(60, rows.filter(r => (r.s - mean) / gap >= 0.65).length)));
  const weak = gap < 0.13;

  // Does plain substring search find these at all?
  const lex = papers.filter(p => q.toLowerCase().split(/\s+/).every(t => p.title.toLowerCase().includes(t))).length;

  console.log(`\n── "${q}"   mean ${mean.toFixed(3)} · gap ${gap.toFixed(3)}${weak ? " (WEAK)" : ""} · shows ${cut} · exact-search finds ${lex}`);
  for (const r of rows.slice(0, 5))
    console.log(`   ${r.s.toFixed(3)} ${String(Math.round(100 * (r.s - mean) / gap)).padStart(3)}  ${r.p.title.slice(0, 86)}`);
}

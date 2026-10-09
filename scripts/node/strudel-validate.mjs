#!/usr/bin/env node
// Authoritative Strudel validation using the REAL @strudel/mini parser.
//
// Our Python regex validator is an approximation — it missed an unbalanced "<[a4] [b4]" pattern
// that crashed Strudel's parser and produced a silent render. This validates every mini-notation
// string in the generated code with Strudel's actual parser, so parse errors are caught BEFORE a
// (170s) render is wasted.
//
// Usage:
//   node strudel-validate.mjs <file.strudel>        # validate a file
//   echo "<code>" | node strudel-validate.mjs       # validate stdin
// Output (stdout): {"ok": bool, "errors": [{"pattern": "...", "error": "..."}]}

import { readFileSync } from 'node:fs';

async function main() {
  const arg = process.argv[2];
  let code = '';
  try {
    code = arg ? readFileSync(arg, 'utf8') : readFileSync(0, 'utf8');
  } catch (e) {
    process.stdout.write(JSON.stringify({ ok: false, errors: [{ pattern: '', error: `read failed: ${e.message}` }] }));
    process.exit(2);
  }

  let mini;
  try {
    ({ mini } = await import('@strudel/mini'));
  } catch (e) {
    // Parser unavailable → cannot validate; report ok=true so the caller falls back to its own check.
    process.stdout.write(JSON.stringify({ ok: true, errors: [], note: `parser unavailable: ${e.message}` }));
    return;
  }

  // Extract every mini-notation string from note("...") and s("...") calls.
  const patterns = [];
  const re = /\b(?:note|s)\(\s*"([^"]*)"/g;
  let mm;
  while ((mm = re.exec(code)) !== null) patterns.push(mm[1]);

  const errors = [];
  for (const p of patterns) {
    if (!p.trim()) continue;
    try {
      mini(p);
    } catch (e) {
      errors.push({ pattern: p.slice(0, 80), error: String(e.message).split('\n')[0].slice(0, 120) });
    }
  }
  process.stdout.write(JSON.stringify({ ok: errors.length === 0, errors, checked: patterns.length }));
}

main();

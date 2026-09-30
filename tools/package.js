// Builds edux-extension.zip (the GitHub Release asset) from the committed EDUX-EXTENSION folder.
// Uses the committed tree, not the working copy, so a release zip never contains uncommitted edits.
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';

const OUT = 'edux-extension.zip';
const ref = process.argv[2] || 'HEAD';

const dirty = execFileSync('git', ['status', '--porcelain', '--', 'EDUX-EXTENSION'], { encoding: 'utf8' }).trim();
if (dirty && ref === 'HEAD') {
  console.warn('⚠️  EDUX-EXTENSION has uncommitted changes; they are NOT included in the zip:\n' + dirty);
}

const manifest = JSON.parse(execFileSync('git', ['show', `${ref}:EDUX-EXTENSION/manifest.json`], { encoding: 'utf8' }));
execFileSync('git', ['archive', '--format=zip', '-o', OUT, `${ref}:EDUX-EXTENSION`]);

const kb = (fs.statSync(OUT).size / 1024).toFixed(0);
console.log(`✓ ${OUT} — v${manifest.version} from ${ref} (${kb} KB)`);

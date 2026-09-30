// Builds edux-extension.zip (the GitHub Release asset) from the committed EDUX-EXTENSION folder.
// Uses the committed tree, not the working copy, so a release zip never contains uncommitted edits.
// Adds files.txt (every file in the zip) so update.ps1 can remove files a later release drops.
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';

const OUT = 'edux-extension.zip';
const ref = process.argv[2] || 'HEAD';
const git = (...args) => execFileSync('git', args, { encoding: 'utf8' });

const dirty = git('status', '--porcelain', '--', 'EDUX-EXTENSION').trim();
if (dirty && ref === 'HEAD') {
  console.warn('⚠️  EDUX-EXTENSION has uncommitted changes; they are NOT included in the zip:\n' + dirty);
}

const manifest = JSON.parse(git('show', `${ref}:EDUX-EXTENSION/manifest.json`));
const files = git('ls-tree', '-r', '--name-only', `${ref}:EDUX-EXTENSION`).trim().split('\n');
const fileList = [...files, 'files.txt'].sort().join('\n') + '\n';

git('archive', '--format=zip', `--add-virtual-file=files.txt:${fileList}`, '-o', OUT, `${ref}:EDUX-EXTENSION`);

const kb = (fs.statSync(OUT).size / 1024).toFixed(0);
console.log(`✓ ${OUT} — v${manifest.version} from ${ref}, ${files.length + 1} files (${kb} KB)`);

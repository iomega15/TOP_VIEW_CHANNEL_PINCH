"""Compare a student's renamed top-view set (zip) against the images already in images_student.

Usage:  python check_renamed_set.py <zip> <H> [results_dir]
Matches files by MD5 (renaming must not change the image), then reports, per design width W:
which membrane-layer (ML) channels the frames cover, overlaps between frames, missing ML values,
and the channel count the pipeline found in each frame (from results_dir/channel_results.csv)
against the count implied by the ML range in the new name. Also lists old files with no renamed twin.
"""
import collections
import csv
import glob
import hashlib
import os
import re
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


def md5(b):
    return hashlib.md5(b).hexdigest()


def main(zpath, H, results=None):
    z = zipfile.ZipFile(zpath)
    new = {md5(z.read(i)): os.path.basename(i.filename) for i in z.infolist() if not i.is_dir()}
    old = {}
    for f in glob.glob(os.path.join(HERE, 'images_student', f'H{H}', '**', '*.jp*g'), recursive=True):
        old[md5(open(f, 'rb').read())] = os.path.relpath(f, HERE)
    counts = collections.Counter()
    if results:
        for r in csv.DictReader(open(os.path.join(HERE, results, 'channel_results.csv'))):
            if r.get('cls') and r['cls'] != 'NO_PORTS_FOUND':
                counts[r['file']] += 1
    print(f'{len(new)} renamed files; {sum(h in old for h in new)} byte-identical to an existing H{H} image')
    cover = collections.defaultdict(list)
    bad = []
    for h, n in sorted(new.items(), key=lambda x: (int(re.search(r'_W(\d+)', x[1]).group(1)), x[1])):
        m = re.search(rf'_H{H}_W(\d+)_ML(\d+)-(\d+)', n)
        if not m:
            bad.append(n)
            continue
        W, a, b = map(int, m.groups())
        cover[W].append((min(a, b), max(a, b)))
        o = os.path.basename(old.get(h, '(no identical old file)'))
        k, found = abs(a - b) + 1, counts.get(o, 0)
        flag = '' if not results or found == k else f'   pipeline found {found}'
        print(f'  {o:48s} -> {n}{flag}')
    for n in bad:
        print('  NAME NOT PARSED:', n)
    print()
    for W in sorted(cover):
        s, dup = set(), set()
        for lo, hi in cover[W]:
            r = set(range(lo, hi + 1))
            dup |= s & r
            s |= r
        miss = sorted(set(range(1, 11)) - s)
        print(f'  W{W}: {len(cover[W])} frame(s), ML {min(s)}-{max(s)}'
              + (f'   MISSING {miss}' if miss else '') + (f'   overlap {sorted(dup)}' if dup else ''))
    left = sorted(os.path.basename(p) for h, p in old.items() if h not in new)
    print(f'\n{len(left)} existing H{H} image(s) with no renamed twin' + (':' if left else ''))
    for p in left:
        print('  ', p)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)

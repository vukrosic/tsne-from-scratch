#!/bin/sh
# Runs the six steps, the tests and the exercise. Expected output: demo-output.txt
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
for f in step1_distances.py step2_keep_distances.py step3_shares.py step4_map.py step5_tail.py step6_lies.py tests.py exercise.py; do
	echo
	echo "=== $f ==="
	$PY $f
done

# submissions/

The submitted predictions contain evidence spans quoted verbatim from posts, so
they are distributed only in the gated bundle.

`make reproduce` rebuilds them and compares every row of every column. The
submitted file `NIDHI_scaled.csv` has MD5 `927debd984bd49fb0b43c691c64bdae4`;
a rebuild written with `python src/build_submission.py --out <file>` matches it
byte for byte.

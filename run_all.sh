for a in bahdanau luong none; do conda run -n precisai python -m src.train --attn $a --epochs 10 2>&1 | grep -E "^(attn|ep|early)"; done

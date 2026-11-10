$ErrorActionPreference = "Continue"
$runs = @(
  @{mem=0; seed=0},
  @{mem=1; seed=0},
  @{mem=0; seed=1},
  @{mem=1; seed=1}
)
foreach ($r in $runs) {
  $out = "results/mem$($r.mem)_s$($r.seed)"
  Write-Output "=== START mem=$($r.mem) seed=$($r.seed) ==="
  python fractal_lm.py --mem $($r.mem) --B 64 --steps 600 --T 128 --seed $($r.seed) --out $out
  Write-Output "=== DONE mem=$($r.mem) seed=$($r.seed) ==="
}

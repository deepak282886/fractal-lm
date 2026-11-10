$ErrorActionPreference = "Continue"
$runs = @(
  @{mem=0; T=256},
  @{mem=1; T=256},
  @{mem=0; T=512},
  @{mem=1; T=512}
)
foreach ($r in $runs) {
  $out = "results/mem$($r.mem)_T$($r.T)"
  $g = if ($r.mem -eq 1) { "--graph 1" } else { "" }
  Write-Output "=== START mem=$($r.mem) T=$($r.T) ==="
  python fractal_lm.py --mem $($r.mem) --B 64 --steps 300 --T $($r.T) --seed 0 --eval_every 300 $g --out $out
  Write-Output "=== DONE mem=$($r.mem) T=$($r.T) ==="
}

$ErrorActionPreference = "Continue"
$runs = @(
  @{mem=1; T=256},
  @{mem=1; T=512}
)
foreach ($r in $runs) {
  $out = "results/mem$($r.mem)_T$($r.T)"
  $a = @('fractal_lm.py','--mem',"$($r.mem)",'--B','64','--steps','300','--T',"$($r.T)",'--seed','0','--eval_every','300','--graph','1','--out',$out)
  Write-Output "=== START mem=$($r.mem) T=$($r.T) ==="
  python @a
  Write-Output "=== DONE mem=$($r.mem) T=$($r.T) ==="
}

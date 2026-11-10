$ErrorActionPreference = "Continue"
$runs = @(
  @{mem=0; T=128; s=0}, @{mem=0; T=128; s=1},
  @{mem=1; T=128; s=0}, @{mem=1; T=128; s=1},
  @{mem=0; T=512; s=0}, @{mem=0; T=512; s=1},
  @{mem=1; T=512; s=0}, @{mem=1; T=512; s=1}
)
foreach ($r in $runs) {
  $out = "results/b_mem$($r.mem)_T$($r.T)_s$($r.s)"
  $a = @('fractal_lm.py','--mem',"$($r.mem)",'--B','64','--steps','3000','--T',"$($r.T)",'--seed',"$($r.s)",'--eval_every','1000','--taus','0.1,0.3,0.5','--out',$out)
  if ($r.mem -eq 1) { $a += @('--graph','1') }
  Write-Output "=== START mem=$($r.mem) T=$($r.T) s=$($r.s) ==="
  python @a
  Write-Output "=== DONE mem=$($r.mem) T=$($r.T) s=$($r.s) ==="
}

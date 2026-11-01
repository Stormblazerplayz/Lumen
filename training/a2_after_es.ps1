# Attempt 2: after the ES run ends, freeze the selected checkpoint and run the pre-registered held-out evaluation once,
# then the hard-scenario diagnostics. Selection is a2_es.py's own (best original-distribution development reward).
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*a2_es.py*' }) { Start-Sleep -Seconds 10 }
Copy-Item a2_es_best.npz a2_final.npz -Force
"final checkpoint frozen at $(Get-Date -Format HH:mm:ss)" | Out-File -Encoding utf8 a2_after_es.log
& py eval2.py --split heldout --arms drive,old,old_unfiltered,new,noshield,control --theta a2_final.npz --workers 5 --out a2_heldout_results.json *> a2_heldout_stdout.log
"held-out done at $(Get-Date -Format HH:mm:ss)" | Out-File -Encoding utf8 -Append a2_after_es.log
& py a2_hard.py a2_final.npz 5 512 *> a2_hard_stdout.log
"hard scenarios done at $(Get-Date -Format HH:mm:ss)" | Out-File -Encoding utf8 -Append a2_after_es.log

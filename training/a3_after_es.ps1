# Attempt 3: after the ES run ends, freeze the selected checkpoint, run the NEW held-out seeds once, then the hard classes.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*a3_es.py*' }) { Start-Sleep -Seconds 10 }
Copy-Item a3_es_best.npz a3_final.npz -Force
"final checkpoint frozen at $(Get-Date -Format HH:mm:ss)" | Out-File -Encoding utf8 a3_after_es.log
& py eval2.py --split heldout3 --arms drive,shipped,a2,new,noshield,control --theta a3_final.npz --workers 5 --out a3_heldout_results.json *> a3_heldout_stdout.log
"held-out done at $(Get-Date -Format HH:mm:ss)" | Out-File -Encoding utf8 -Append a3_after_es.log
& py a3_hard.py a3_final.npz 5 512 *> a3_hard_stdout.log
"hard scenarios done at $(Get-Date -Format HH:mm:ss)" | Out-File -Encoding utf8 -Append a3_after_es.log

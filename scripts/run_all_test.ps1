# Final TEST-split runs, two parallel worker chains (each chain owns distinct memory configurations).
$py = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$root = Join-Path $PSScriptRoot ".."
Set-Location $root
$d = "data\synthetic\main"
$chain1 = "& '$py' scripts\run_experiment.py --data $d --methods E_agentmamba,H_learned_mamba,A_no_retrieval,D_learned_always,E_no_verification,E_first_candidate,G_learned_nomamba,E_mlp_gate; " +
          "& '$py' scripts\run_experiment.py --data $d --sweep --sweep-policies learned_mamba,learned_nomamba"
$chain2 = "& '$py' scripts\run_experiment.py --data $d --methods B_storeall_rag,B_storeall_no_verify,B_storeall_vector_only,B_storeall_conditional,C_heuristic_rag,F_random_rag; " +
          "& '$py' scripts\run_experiment.py --data $d --sweep --sweep-policies heuristic,random,store_all"
Start-Process powershell -ArgumentList "-NoProfile", "-Command", "$chain1 2> logs\test_chain1.err" -WindowStyle Hidden
Start-Process powershell -ArgumentList "-NoProfile", "-Command", "$chain2 2> logs\test_chain2.err" -WindowStyle Hidden

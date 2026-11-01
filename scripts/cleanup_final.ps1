# EgoHack Final Cleanup Script
# CAUTION: This script permanently deletes hackathon data, outputs, and cloud resources.
# It requires typing YES before each group to prevent accidental deletion.

[CmdletBinding()]
param()

function Confirm-Action($promptMsg) {
    Write-Host ""
    Write-Host "=======================================================" -ForegroundColor Yellow
    Write-Host "$promptMsg" -ForegroundColor Cyan
    Write-Host "=======================================================" -ForegroundColor Yellow
    $response = Read-Host "Type 'YES' to delete, or press Enter to skip"
    return ($response -ceq "YES")
}

$egohackRoot = "C:\Users\kathu\OneDrive\Desktop\egohack"
$v2Root = "C:\Users\kathu\egohack_v2"
$kaggleCli = "$egohackRoot\egohack\Scripts\kaggle.exe"

Write-Host "=== EGOHACK POST-HACKATHON SAFE CLEANUP WIZARD ===" -ForegroundColor Green
Write-Host "Each group will require typing YES in ALL CAPS to proceed.`n"

# Group 1: External storage drives (C:\egohack_data, D:\egohack_data)
if (Confirm-Action "Group 1: Delete external drive data (C:\egohack_data, D:\egohack_data)?") {
    @("C:\egohack_data", "D:\egohack_data") | ForEach-Object {
        if (Test-Path $_) {
            Write-Host "Deleting $_..." -ForegroundColor Red
            Remove-Item -Recurse -Force $_ -ErrorAction SilentlyContinue
        } else {
            Write-Host "$_ does not exist. Skipped." -ForegroundColor Gray
        }
    }
} else { Write-Host "Group 1 skipped." -ForegroundColor Yellow }

# Group 2: Project data folders (data/)
if (Confirm-Action "Group 2: Delete local data folders ($egohackRoot\data, $v2Root\data)?") {
    @("$egohackRoot\data", "$v2Root\data") | ForEach-Object {
        if (Test-Path $_) {
            Write-Host "Deleting $_..." -ForegroundColor Red
            Remove-Item -Recurse -Force $_ -ErrorAction SilentlyContinue
        } else {
            Write-Host "$_ does not exist. Skipped." -ForegroundColor Gray
        }
    }
} else { Write-Host "Group 2 skipped." -ForegroundColor Yellow }

# Group 3: Output folders (out/)
if (Confirm-Action "Group 3: Delete pipeline out folders ($egohackRoot\out, $v2Root\out)?") {
    @("$egohackRoot\out", "$v2Root\out") | ForEach-Object {
        if (Test-Path $_) {
            Write-Host "Deleting $_..." -ForegroundColor Red
            Remove-Item -Recurse -Force $_ -ErrorAction SilentlyContinue
        } else {
            Write-Host "$_ does not exist. Skipped." -ForegroundColor Gray
        }
    }
} else { Write-Host "Group 3 skipped." -ForegroundColor Yellow }

# Group 4: Eval frames & montages (eval/ai_frames, eval/montage, dev_frames, eval/*.png)
if (Confirm-Action "Group 4: Delete eval frames and montages (ai_frames, montage, dev_frames, *.png)?") {
    @("$egohackRoot\eval\ai_frames", "$v2Root\eval\ai_frames",
      "$egohackRoot\eval\montage", "$v2Root\eval\montage",
      "$egohackRoot\dev_frames", "$v2Root\dev_frames") | ForEach-Object {
        if (Test-Path $_) {
            Write-Host "Deleting $_..." -ForegroundColor Red
            Remove-Item -Recurse -Force $_ -ErrorAction SilentlyContinue
        }
    }
    Get-ChildItem -Path @("$egohackRoot\eval", "$v2Root\eval") -Include *.png, *.jpg -Recurse -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
    Write-Host "Eval frames cleaned." -ForegroundColor Green
} else { Write-Host "Group 4 skipped." -ForegroundColor Yellow }

# Group 5: Kaggle upload folders (kaggle_upload/)
if (Confirm-Action "Group 5: Delete local kaggle_upload folders?") {
    @("$egohackRoot\kaggle_upload", "$v2Root\kaggle_upload") | ForEach-Object {
        if (Test-Path $_) {
            Write-Host "Deleting $_..." -ForegroundColor Red
            Remove-Item -Recurse -Force $_ -ErrorAction SilentlyContinue
        }
    }
} else { Write-Host "Group 5 skipped." -ForegroundColor Yellow }

# Group 6: Weights & MANO models (weights/, mano/)
if (Confirm-Action "Group 6: Delete model weights and MANO models (weights/, mano/)?") {
    @("$egohackRoot\weights", "$v2Root\weights",
      "$egohackRoot\mano", "$v2Root\mano") | ForEach-Object {
        if (Test-Path $_) {
            Write-Host "Deleting $_..." -ForegroundColor Red
            Remove-Item -Recurse -Force $_ -ErrorAction SilentlyContinue
        }
    }
} else { Write-Host "Group 6 skipped." -ForegroundColor Yellow }

# Group 7: External detector checkpoint (361 MB) & hands_ext files
if (Confirm-Action "Group 7: Delete external detector 361 MB checkpoint, output runs & hands_ext.jsonl?") {
    @("$v2Root\hands_ext.jsonl", "$egohackRoot\hands_ext.jsonl",
      "$v2Root\external\output_complete", "$v2Root\external\output_run1", "$v2Root\external\output_run2") | ForEach-Object {
        if (Test-Path $_) {
            Write-Host "Deleting $_..." -ForegroundColor Red
            Remove-Item -Recurse -Force $_ -ErrorAction SilentlyContinue
        }
    }
} else { Write-Host "Group 7 skipped." -ForegroundColor Yellow }

# Group 8: Remote Kaggle resources (via Kaggle CLI)
if (Confirm-Action "Group 8: Delete private Kaggle Dataset & Kernel (via kaggle CLI)?") {
    if (Test-Path $kaggleCli) {
        Write-Host "Deleting Kaggle Kernel: pulkit321/egohack-ext-baseline..." -ForegroundColor Red
        & $kaggleCli kernels delete pulkit321/egohack-ext-baseline -y -ErrorAction SilentlyContinue
        Write-Host "Deleting Kaggle Dataset: pulkit321/egohack-tune-100..." -ForegroundColor Red
        & $kaggleCli datasets delete pulkit321/egohack-tune-100 -y -ErrorAction SilentlyContinue
    } else {
        Write-Host "kaggle CLI not found at $kaggleCli. Delete manually on kaggle.com." -ForegroundColor Yellow
    }
} else { Write-Host "Group 8 skipped." -ForegroundColor Yellow }

# Group 9: Windows Recycle Bin
if (Confirm-Action "Group 9: Empty Windows Recycle Bin?") {
    Clear-RecycleBin -Force -ErrorAction SilentlyContinue
    Write-Host "Recycle Bin emptied." -ForegroundColor Green
} else { Write-Host "Group 9 skipped." -ForegroundColor Yellow }

Write-Host "`n=== CLEANUP WIZARD FINISHED ===" -ForegroundColor Green


<#
.SYNOPSIS
    Runs specific ablation study experiments.

.DESCRIPTION
    This script automates the execution of the ablation study experiments defined in docs/ablation_study_design.md.
    It passes the --experiment_name flag to ensure results are saved in dedicated folders.

.PARAMETER Id
    The Experiment ID to run (e.g., UNET-A, DL-C, SF-B).
    
.EXAMPLE
    .\tools\run_ablation.ps1 -Id UNET-A
#>

param (
    [Parameter(Mandatory = $true)]
    [ValidateSet("UNET-A", "UNET-B", "UNET-C", "DL-A", "DL-B", "DL-C", "SF-A", "SF-B", "DL-D")]
    [string]$Id
)

$ErrorActionPreference = "Stop"

function Run-Command ($cmd) {
    Write-Host "Executing: $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
}

Write-Host "Starting Experiment: $Id" -ForegroundColor Green

# Common flags can be added here if needed
$ExpFlag = "--experiment_name $Id"

switch ($Id) {
    "UNET-A" {
        # Baseline Unweighted
        Run-Command "python -m src.train --model unet --epochs 20 $ExpFlag"
    }
    "UNET-B" {
        # Weighted
        Run-Command "python -m src.train --model unet --epochs 20 --weighted_loss $ExpFlag"
    }
    "UNET-C" {
        # Mixed: Weighted (10) -> Unweighted (10)
        # 1. Run Weighted
        Run-Command "python -m src.train --model unet --epochs 10 --weighted_loss $ExpFlag"
        # 2. Resume Unweighted
        $Checkpoint = "checkpoints/${Id}_latest.pth"
        if (Test-Path $Checkpoint) {
            Run-Command "python -m src.train --model unet --epochs 10 --resume $Checkpoint $ExpFlag"
        }
        else {
            Write-Error "Checkpoint not found: $Checkpoint"
        }
    }
    
    "DL-A" {
        # Naive Unfrozen
        Run-Command "python -m src.train --model deeplab --epochs 20 --weighted_loss $ExpFlag"
    }
    "DL-B" {
        # Frozen Only
        Run-Command "python -m src.train --model deeplab --epochs 20 --weighted_loss --freeze_backbone $ExpFlag"
    }
    "DL-C" {
        # Two-Stage: Frozen (10) -> Unfrozen (10)
        # 1. Run Frozen
        Run-Command "python -m src.train --model deeplab --epochs 10 --weighted_loss --freeze_backbone $ExpFlag"
        # 2. Resume Unfrozen
        $Checkpoint = "checkpoints/${Id}_latest.pth"
        if (Test-Path $Checkpoint) {
            Run-Command "python -m src.train --model deeplab --epochs 10 --weighted_loss --resume $Checkpoint $ExpFlag"
        }
        else {
            Write-Error "Checkpoint not found: $Checkpoint"
        }
    }
    "DL-D" {
        # Full Scale Two-Stage
        # 1. Run Frozen Full Scale
        Run-Command "python -m src.train --model deeplab --full_scale --epochs 10 --weighted_loss --freeze_backbone $ExpFlag"
        # 2. Resume Unfrozen Full Scale
        $Checkpoint = "checkpoints/${Id}_latest.pth"
        if (Test-Path $Checkpoint) {
            Run-Command "python -m src.train --model deeplab --full_scale --epochs 10 --weighted_loss --resume $Checkpoint $ExpFlag"
        }
        else {
            Write-Error "Checkpoint not found: $Checkpoint"
        }
    }

    "SF-A" {
        # Baseline Cropped (Two-Stage assumed)
        Run-Command "python -m src.train --model segformer --epochs 10 --weighted_loss --freeze_backbone $ExpFlag"
        $Checkpoint = "checkpoints/${Id}_latest.pth"
        if (Test-Path $Checkpoint) {
            Run-Command "python -m src.train --model segformer --epochs 10 --weighted_loss --resume $Checkpoint $ExpFlag"
        }
    }
    "SF-B" {
        # Full Scale (Two-Stage)
        Run-Command "python -m src.train --model segformer --full_scale --epochs 10 --weighted_loss --freeze_backbone $ExpFlag"
        $Checkpoint = "checkpoints/${Id}_latest.pth"
        if (Test-Path $Checkpoint) {
            Run-Command "python -m src.train --model segformer --full_scale --epochs 10 --weighted_loss --resume $Checkpoint $ExpFlag"
        }
    }
}

Write-Host "Experiment $Id Completed Successfully." -ForegroundColor Green

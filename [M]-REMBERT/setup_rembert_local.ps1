# RemBERT LOCAL SETUP - AUTOMATED INSTALLATION SCRIPT
# Usage: .\setup_rembert_local.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  RemBERT Local Training Setup Script  " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "[1/7] Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "  Found: $pythonVersion" -ForegroundColor Green

if ($pythonVersion -match "Python 3\.13") {
    Write-Host "  WARNING: Python 3.13 may have compatibility issues!" -ForegroundColor Red
    Write-Host "  Recommended: Python 3.12 or 3.11" -ForegroundColor Yellow
    $continue = Read-Host "  Continue anyway? (y/n)"
    if ($continue -ne "y") {
        Write-Host "  Setup cancelled. Please install Python 3.12 or 3.11." -ForegroundColor Red
        exit
    }
}

# Check CUDA availability
Write-Host ""
Write-Host "[2/7] Checking NVIDIA GPU..." -ForegroundColor Yellow
$nvidiaSmi = nvidia-smi 2>$null
if ($?) {
    Write-Host "  GPU detected!" -ForegroundColor Green
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | ForEach-Object {
        Write-Host "  $_" -ForegroundColor Green
    }
} else {
    Write-Host "  WARNING: No NVIDIA GPU detected!" -ForegroundColor Red
    Write-Host "  Training will be VERY slow on CPU." -ForegroundColor Yellow
    $continue = Read-Host "  Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit
    }
}

# Create virtual environment
Write-Host ""
Write-Host "[3/7] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv_rembert") {
    Write-Host "  Virtual environment already exists." -ForegroundColor Yellow
    $recreate = Read-Host "  Recreate? (y/n)"
    if ($recreate -eq "y") {
        Remove-Item -Recurse -Force venv_rembert
        python -m venv venv_rembert
        Write-Host "  Virtual environment recreated." -ForegroundColor Green
    }
} else {
    python -m venv venv_rembert
    Write-Host "  Virtual environment created." -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "[4/7] Activating virtual environment..." -ForegroundColor Yellow
& ".\venv_rembert\Scripts\Activate.ps1"
if ($?) {
    Write-Host "  Virtual environment activated." -ForegroundColor Green
} else {
    Write-Host "  ERROR: Failed to activate virtual environment." -ForegroundColor Red
    Write-Host "  You may need to change execution policy:" -ForegroundColor Yellow
    Write-Host "  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    exit
}

# Install PyTorch with CUDA
Write-Host ""
Write-Host "[5/7] Installing PyTorch with CUDA 12.1..." -ForegroundColor Yellow
Write-Host "  This may take several minutes..." -ForegroundColor Cyan
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121 --quiet

if ($?) {
    Write-Host "  PyTorch installed successfully." -ForegroundColor Green
    
    # Verify CUDA
    $cudaCheck = python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>&1
    Write-Host "  $cudaCheck" -ForegroundColor Green
} else {
    Write-Host "  ERROR: PyTorch installation failed." -ForegroundColor Red
    exit
}

# Install other dependencies
Write-Host ""
Write-Host "[6/7] Installing other dependencies..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes..." -ForegroundColor Cyan
pip install -r requirements_rembert.txt --quiet

if ($?) {
    Write-Host "  All dependencies installed successfully." -ForegroundColor Green
} else {
    Write-Host "  WARNING: Some dependencies may have failed." -ForegroundColor Yellow
}

# Verify installation
Write-Host ""
Write-Host "[7/7] Verifying installation..." -ForegroundColor Yellow

# Test imports
$testScript = @"
import sys
import numpy as np
import pandas as pd
import torch
import transformers
from packaging import version

print('Python:', sys.version.split()[0])
print('NumPy:', np.__version__)
print('Pandas:', pd.__version__)
print('PyTorch:', torch.__version__)
print('Transformers:', transformers.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU Memory:', f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')

# Test RemBERT tokenizer download
print('\nTesting RemBERT model access...')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('google/rembert')
print('✓ RemBERT tokenizer loaded successfully!')
"@

$testScript | python
if ($?) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  Setup Complete!                       " -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Review LOCAL_REMBERT_SETUP_GUIDE.md" -ForegroundColor White
    Write-Host "  2. Create RemBERT notebook" -ForegroundColor White
    Write-Host "  3. Update paths in Section 3" -ForegroundColor White
    Write-Host "  4. Run training!" -ForegroundColor White
    Write-Host ""
    Write-Host "To activate environment in future:" -ForegroundColor Yellow
    Write-Host "  .\venv_rembert\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  Setup Incomplete                      " -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Some components failed to load." -ForegroundColor Yellow
    Write-Host "Check error messages above." -ForegroundColor Yellow
    Write-Host ""
}

@echo off
cls
echo.
echo ================================================
echo   SOCIAL SCALPING - QUANTUM TRADING SYSTEM
echo ================================================
echo.
echo ATTENTION: Ce programme est TRES lourd !
echo - 5,787 lignes de code
echo - TensorFlow, XGBoost, LightGBM requis
echo - ~1 GB de RAM minimum
echo.
echo Demarrage...
echo.

python SOCIAL_SCALPING.py

if %errorlevel% neq 0 (
    echo.
    echo [ERREUR] Le programme a plante !
    echo.
    echo Dependances manquantes ?
    echo.
    echo Installation:
    echo pip install MetaTrader5 numpy pandas scipy scikit-learn
    echo pip install xgboost lightgbm tensorflow joblib
    echo.
    echo ATTENTION: TA-Lib necessite installation speciale sur Windows
    echo Telecharger depuis: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
    echo.
    pause
)

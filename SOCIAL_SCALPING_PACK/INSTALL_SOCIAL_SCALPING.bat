@echo off
cls
echo.
echo ================================================
echo   INSTALLATION SOCIAL_SCALPING
echo ================================================
echo.
echo Ce script va installer TOUTES les dependances
echo Cela peut prendre 10-15 minutes !
echo.
pause
echo.

echo [1/8] Installation MetaTrader5...
pip install MetaTrader5

echo.
echo [2/8] Installation NumPy...
pip install numpy

echo.
echo [3/8] Installation Pandas...
pip install pandas

echo.
echo [4/8] Installation SciPy...
pip install scipy

echo.
echo [5/8] Installation Scikit-learn...
pip install scikit-learn

echo.
echo [6/8] Installation XGBoost...
pip install xgboost

echo.
echo [7/8] Installation LightGBM...
pip install lightgbm

echo.
echo [8/8] Installation TensorFlow...
pip install tensorflow

echo.
echo Installation Joblib...
pip install joblib

echo.
echo ================================================
echo   INSTALLATION TERMINEE !
echo ================================================
echo.
echo ATTENTION: TA-Lib necessite installation manuelle
echo.
echo 1. Telechargez le fichier .whl depuis:
echo    https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
echo.
echo 2. Choisissez le fichier correspondant a votre Python:
echo    - Python 3.11 64-bit: TA_Lib-0.4.XX-cp311-cp311-win_amd64.whl
echo.
echo 3. Installez avec:
echo    pip install TA_Lib-0.4.XX-cp311-cp311-win_amd64.whl
echo.
echo Une fois TA-Lib installe, lancez START_SOCIAL_SCALPING.bat
echo.
pause

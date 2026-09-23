@echo off
rem Abre o painel do HidroVision AI no navegador. Basta dar dois cliques.
cd /d "%~dp0"

rem mesmo interpretador do rodar.ps1; o "python" do PATH pode ser outro, sem o streamlit
set "PYTHON=C:\Users\yande\AppData\Local\Python\pythoncore-3.14-64\python.exe"

rem credenciais da ANA ficam fora do repositorio (modelo em credenciais.bat.exemplo)
if exist "credenciais.bat" (
    call "credenciais.bat"
) else (
    echo credenciais.bat nao encontrado: o painel abre, mas nao consulta a ANA.
)

"%PYTHON%" -m streamlit run "integracao\dashboard.py" --browser.gatherUsageStats false

pause

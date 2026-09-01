@echo off
setlocal

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"

set "VCVARSALL="
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        if exist "%%i\VC\Auxiliary\Build\vcvarsall.bat" set "VCVARSALL=%%i\VC\Auxiliary\Build\vcvarsall.bat"
    )
)

if "%VCVARSALL%"=="" (
    if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VCVARSALL=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
    ) else if exist "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VCVARSALL=C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
    )
)

if not "%VCVARSALL%"=="" (
    echo Initializing Visual Studio environment from %VCVARSALL% for architecture: %WINDOWS_VC_TARGET%
    call "%VCVARSALL%" %WINDOWS_VC_TARGET%
)

echo Building GLPK for architecture: %WINDOWS_VC_TARGET% and GLP target %WINDOWS_GLPK_TARGET%

pushd glpk\%WINDOWS_GLPK_TARGET%

copy config_VC config.h

nmake /f Makefile_VC glpk.lib
if errorlevel 1 (
    echo Error: nmake failed
    popd
    exit /b 1
)

if not exist C:\glpk-install\lib md C:\glpk-install\lib
if not exist C:\glpk-install\include md C:\glpk-install\include

copy glpk.lib C:\glpk-install\lib\glpk.lib
copy ..\src\glpk.h C:\glpk-install\include\

dir C:\glpk-install\include
dir C:\glpk-install\lib

popd

if not exist src\python\.libs md src\python\.libs
if exist C:\glpk-install\bin\*.dll copy C:\glpk-install\bin\*.dll src\python\.libs\
if exist C:\glpk-install\lib\*.dll copy C:\glpk-install\lib\*.dll src\python\.libs\

endlocal

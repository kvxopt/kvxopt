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

echo Building GSL for architecture: %WINDOWS_CMAKE_TARGET%

git clone --recursive https://github.com/ampl/gsl.git

pushd gsl

git checkout %GSL_COMMIT_HASH%

md build
pushd build

cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -A %WINDOWS_CMAKE_TARGET% -DBUILD_SHARED_LIBS=ON -DMSVC_RUNTIME_DYNAMIC=ON -DNO_AMPL_BINDINGS=ON -DGSL_DISABLE_TESTS=ON -DCMAKE_INSTALL_PREFIX="C:/gsl-install" ..

cmake --build . --config Release --target install

popd
popd

if not exist src\python\.libs md src\python\.libs
if exist C:\gsl-install\bin\*.dll copy C:\gsl-install\bin\*.dll src\python\.libs\
if exist C:\gsl-install\bin\Release\*.dll copy C:\gsl-install\bin\Release\*.dll src\python\.libs\
if exist C:\gsl-install\lib\*.dll copy C:\gsl-install\lib\*.dll src\python\.libs\
if exist C:\gsl-install\lib\Release\*.dll copy C:\gsl-install\lib\Release\*.dll src\python\.libs\

endlocal

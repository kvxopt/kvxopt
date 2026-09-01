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

echo Building FFTW for architecture: %WINDOWS_CMAKE_TARGET%

pushd fftw-%FFTW_VERSION%

md build
pushd build

cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -A %WINDOWS_CMAKE_TARGET% -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DENABLE_THREADS=ON -DWITH_COMBINED_THREADS=ON -DCMAKE_INSTALL_PREFIX="C:/fftw-install" ..

cmake --build . --config Release --target install

popd
popd

if not exist src\python\.libs md src\python\.libs
if exist C:\fftw-install\bin\*.dll copy C:\fftw-install\bin\*.dll src\python\.libs\
if exist C:\fftw-install\bin\Release\*.dll copy C:\fftw-install\bin\Release\*.dll src\python\.libs\
if exist C:\fftw-install\lib\*.dll copy C:\fftw-install\lib\*.dll src\python\.libs\
if exist C:\fftw-install\lib\Release\*.dll copy C:\fftw-install\lib\Release\*.dll src\python\.libs\

echo Building FFTW completed successfully.

dir C:\fftw-install\include
dir C:\fftw-install\lib

endlocal

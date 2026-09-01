setlocal enabledelayedexpansion

REM Set default architecture if not provided
if "%ARCH%"=="" (
    set ARCH=x64
)

if "%OSTYPE%"=="" (
    set OSTYPE=win32
)

echo Installing Windows dependencies for architecture: %ARCH%

if not exist src\python\.libs md src\python\.libs

if "%KVXOPT_BUILD_GLPK%"=="1" (
    echo Building GLPK
    REM Download and build GLPK
    call .ci\library_builders\download_glpk.bat || exit /b 1
    call .ci\library_builders\compile_glpk.bat || exit /b 1
) else (
    echo Skipping GLPK build
)

if "%KVXOPT_BUILD_FFTW%"=="1" (
    echo Building FFTW
    REM Download and prepare FFTW binaries
    bash .ci\library_builders\download_fftw.sh || exit /b 1
    call .ci\library_builders\prepare_fftw.bat || exit /b 1
) else (
    echo Skipping FFTW build
)

REM Download OpenBLAS binaries
bash .ci\library_builders\download_openblas.sh || exit /b 1

REM Download SuiteSparse source
bash .ci/library_builders/download_suitesparse.sh || exit /b 1

if "%KVXOPT_BUILD_OSQP%"=="1" (
    echo Building OSQP
    REM Download and build OSQP
    bash .ci\library_builders\build_osqp.sh || exit /b 1
) else (
    echo Skipping OSQP build
)

if "%KVXOPT_BUILD_GSL%"=="1" (
    echo Building GSL
    REM Download and build GSL
    call .ci\library_builders\compile_gsl.bat || exit /b 1
) else (
    echo Skipping GSL build
)

echo Windows dependencies installed successfully for %ARCH%

endlocal

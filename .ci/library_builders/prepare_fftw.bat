echo Initializing Visual Studio environment for architecture: %WINDOWS_VC_TARGET%

call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" %WINDOWS_VC_TARGET%

echo Building FFTW for architecture: %WINDOWS_CMAKE_TARGET%

pushd fftw-%FFTW_VERSION%

md build
pushd build

cmake -G "Visual Studio 17 2022" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -A %WINDOWS_CMAKE_TARGET% -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DENABLE_THREADS=ON -DWITH_COMBINED_THREADS=ON -DCMAKE_INSTALL_PREFIX="C:/fftw-install" ..

cmake --build . --config Release --target install

popd
popd

if not exist src\python\.libs md src\python\.libs
if exist C:\fftw-install\bin\*.dll copy C:\fftw-install\bin\*.dll src\python\.libs\
if exist C:\fftw-install\lib\*.dll copy C:\fftw-install\lib\*.dll src\python\.libs\

echo Building FFTW completed successfully.

dir C:\fftw-install\include
dir C:\fftw-install\lib

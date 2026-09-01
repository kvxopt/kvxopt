#!/bin/bash
set -e

# Source configuration
if [ -f ".ci/config/versions.env" ]; then
    source .ci/config/versions.env
fi

echo "Downloading FFTW ${FFTW_VERSION} for Windows..."

# Download using curl
curl -L -o "fftw-${FFTW_VERSION}.tar.gz" \
    "https://www.fftw.org/pub/fftw/fftw-${FFTW_VERSION}.tar.gz"

# Verify download was successful
if [ ! -f "fftw-${FFTW_VERSION}.tar.gz" ]; then
    echo "Error: FFTW download failed"
    exit 1
fi

# Get file size to ensure it's not empty
FILE_SIZE=$(stat -c%s "fftw-${FFTW_VERSION}.tar.gz" 2>/dev/null || echo 0)
if [ "$FILE_SIZE" -lt 1000 ]; then
    echo "Error: Downloaded file is too small (${FILE_SIZE} bytes)"
    exit 1
fi

# Verify SHA256 checksum using certutil (on Windows) or shasum/sha256sum
echo "Verifying SHA256 checksum..."
if command -v certutil &> /dev/null; then
    ACTUAL_HASH=$(certutil -hashfile "fftw-${FFTW_VERSION}.tar.gz" SHA256 | grep -v "SHA256" | grep -v "CertUtil" | tr -d ' \r\n' | tr '[:upper:]' '[:lower:]')
elif command -v sha256sum &> /dev/null; then
    ACTUAL_HASH=$(sha256sum "fftw-${FFTW_VERSION}.tar.gz" | awk '{print $1}')
else
    ACTUAL_HASH=$(shasum -a 256 "fftw-${FFTW_VERSION}.tar.gz" | awk '{print $1}')
fi

EXPECTED_HASH=$(echo "${FFTW_SHA256}" | tr '[:upper:]' '[:lower:]')

echo "Expected: ${EXPECTED_HASH}"
echo "Actual:   ${ACTUAL_HASH}"

if [ "${ACTUAL_HASH}" != "${EXPECTED_HASH}" ]; then
    echo "Error: SHA256 checksum mismatch!"
    echo "Expected: ${EXPECTED_HASH}"
    echo "Actual:   ${ACTUAL_HASH}"
    exit 1
else
    echo "SHA256 verification passed"
fi

# Extract using tar
echo "Extracting FFTW..."
tar -xf "fftw-${FFTW_VERSION}.tar.gz"

# Clean up
rm "fftw-${FFTW_VERSION}.tar.gz"

echo "FFTW setup completed"

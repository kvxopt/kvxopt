pipeline {
  agent none
  environment {
    KVXOPT_BUILD_GSL = '1'
    KVXOPT_BUILD_FFTW = '1'
    KVXOPT_BUILD_GLPK = '1'
    KVXOPT_BUILD_DSDP = '1'
    KVXOPT_BUILD_OSQP = '1'
    SUITESPARSE_VERSION = '7.14.0'
    SUITESPARSE_SHA256 = 'c552c4b4bb7d0978796e57263a73295bca0c6b41ad137b45b4f264cfe9300fcb'
    OSQP_VERSION = '0.6.2'
    OSQP_SHA256 = '0a7ade2fa19f13e13bc12f6ea0046ef764049023efb4997a4e72a76534f623ec'
    PREFIX_LINUX = '/usr/local'
  }
  stages {
    stage('Linux build') {
      parallel {
        stage ('Test in docker') {
          agent { docker { image 'python:3' } }
          stages {

            stage ('Set environment paths') {
              steps {
                script {
                  env.PREFIX = "${PREFIX_LINUX}"
                  env.KVXOPT_OSQP_LIB_DIR = "${PREFIX}/lib"
                  env.KVXOPT_OSQP_INC_DIR = "${PREFIX}/include/osqp"
                  env.LD_LIBRARY_PATH = "${PREFIX}/lib"
                }
              }
            }

            stage('Install python dependencies') {
              steps {
                sh '''python -m pip install --upgrade pip
                      pip install --upgrade setuptools setuptools_scm build wheel pytest pytest-cov coveralls numpy'''
              }
            }

            stage('Install libraries') {
              steps {
                sh '''apt-get update'''
                sh '''DEBIAN_FRONTEND=noninteractive apt-get -yq install libopenblas-dev libfftw3-dev libglpk-dev libdsdp-dev libgsl0-dev libsuitesparse-dev'''
              }
            }

            stage('Install cmake') {
              steps {
                sh '''apt-get update
                      DEBIAN_FRONTEND=noninteractive apt-get -yq install cmake'''
              }
            }

            stage('Build OSQP') {
              steps {
                sh '''git clone --recursive https://github.com/oxfordcontrol/osqp.git
                      cd osqp
                      git checkout v${OSQP_VERSION}
                      mkdir build
                      cd build
                      cmake -DCMAKE_INSTALL_PREFIX="${PREFIX}" -DCMAKE_BUILD_TYPE=Release ..
                      make
                      make install
                      cd ../..
                      rm -rf osqp'''
              }
            }

            stage('Build and install kvxopt') {
              steps {
                sh '''python -m pip install .'''
              }
            }

            stage('Test') {
              steps {
                sh '''pytest --cov=kvxopt -s'''
              }
            }
          }
        }
      }
    }
  }
}

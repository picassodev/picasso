ARG BASE=nvidia/cuda:10.2-devel
FROM $BASE

ARG NPROCS=4

RUN apt-get update && apt-get install -y \
        bc \
        wget \
        openssh-client \
        jq \
        git \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN KEYDUMP_URL=https://cloud.cees.ornl.gov/download && \
    KEYDUMP_FILE=keydump && \
    wget --quiet ${KEYDUMP_URL}/${KEYDUMP_FILE} && \
    wget --quiet ${KEYDUMP_URL}/${KEYDUMP_FILE}.sig && \
    gpg --import ${KEYDUMP_FILE} && \
    gpg --verify ${KEYDUMP_FILE}.sig ${KEYDUMP_FILE} && \
    rm ${KEYDUMP_FILE}*

# Install CMake
ENV CMAKE_DIR=/opt/cmake
RUN CMAKE_VERSION=3.18.5 && \
    CMAKE_KEY=2D2CEF1034921684 && \
    CMAKE_URL=https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION} && \
    CMAKE_SCRIPT=cmake-${CMAKE_VERSION}-Linux-x86_64.sh && \
    CMAKE_SHA256=cmake-${CMAKE_VERSION}-SHA-256.txt && \
    wget --quiet ${CMAKE_URL}/${CMAKE_SHA256} && \
    wget --quiet ${CMAKE_URL}/${CMAKE_SHA256}.asc && \
    wget --quiet ${CMAKE_URL}/${CMAKE_SCRIPT} && \
    gpg --verify ${CMAKE_SHA256}.asc ${CMAKE_SHA256} && \
    grep ${CMAKE_SCRIPT} ${CMAKE_SHA256} | sha256sum --check && \
    mkdir -p ${CMAKE_DIR} && \
    sh ${CMAKE_SCRIPT} --skip-license --prefix=${CMAKE_DIR} && \
    rm cmake*
ENV PATH=${CMAKE_DIR}/bin:$PATH

# Install CUDA-aware Open MPI
ENV OPENMPI_DIR=/opt/openmpi
RUN OPENMPI_VERSION=4.0.2 && \
    OPENMPI_VERSION_SHORT=4.0 && \
    OPENMPI_SHA1=32ce3761288575fb8e4f6296c9105c3a25cf3235 && \
    OPENMPI_URL=https://download.open-mpi.org/release/open-mpi/v${OPENMPI_VERSION_SHORT}/openmpi-${OPENMPI_VERSION}.tar.bz2 && \
    OPENMPI_ARCHIVE=openmpi-${OPENMPI_VERSION}.tar.bz2 && \
    [ ! -z "${CUDA_VERSION}" ] && CUDA_OPTIONS=--with-cuda || true && \
    SCRATCH_DIR=/scratch && mkdir -p ${SCRATCH_DIR} && cd ${SCRATCH_DIR} && \
    wget --quiet ${OPENMPI_URL} --output-document=${OPENMPI_ARCHIVE} && \
    echo "${OPENMPI_SHA1} ${OPENMPI_ARCHIVE}" | sha1sum -c && \
    mkdir -p openmpi && \
    tar -xf ${OPENMPI_ARCHIVE} -C openmpi --strip-components=1 && \
    mkdir -p build && cd build && \
    ../openmpi/configure --prefix=${OPENMPI_DIR} ${CUDA_OPTIONS} CFLAGS=-w && \
    make -j${NPROCS} install && \
    rm -rf ${SCRATCH_DIR}
ENV PATH=${OPENMPI_DIR}/bin:$PATH

# Install Boost
ENV BOOST_DIR=/opt/boost
RUN BOOST_VERSION=1.75.0 && \
    BOOST_VERSION_UNDERSCORE=$(echo "$BOOST_VERSION" | sed -e "s/\./_/g") && \
    BOOST_KEY=379CE192D401AB61 && \
    BOOST_URL=https://boostorg.jfrog.io/artifactory/main/release/${BOOST_VERSION}/source && \
    BOOST_ARCHIVE=boost_${BOOST_VERSION_UNDERSCORE}.tar.bz2 && \
    SCRATCH_DIR=/scratch && mkdir -p ${SCRATCH_DIR} && cd ${SCRATCH_DIR} && \
    wget --quiet ${BOOST_URL}/${BOOST_ARCHIVE} && \
    wget --quiet ${BOOST_URL}/${BOOST_ARCHIVE}.asc && \
    wget --quiet ${BOOST_URL}/${BOOST_ARCHIVE}.json && \
    wget --quiet ${BOOST_URL}/${BOOST_ARCHIVE}.json.asc && \
    gpg --verify ${BOOST_ARCHIVE}.json.asc ${BOOST_ARCHIVE}.json && \
    gpg --verify ${BOOST_ARCHIVE}.asc ${BOOST_ARCHIVE} && \
    cat ${BOOST_ARCHIVE}.json | jq -r '. | .sha256 + "  " + .file' | sha256sum --check && \
    mkdir -p boost && \
    tar -xf ${BOOST_ARCHIVE} -C boost --strip-components=1 && \
    cd boost && \
    ./bootstrap.sh \
        --prefix=${BOOST_DIR} \
        && \
    echo "using mpi ;" >> project-config.jam && \
    ./b2 -j${NPROCS} \
        hardcode-dll-paths=true dll-path=${BOOST_DIR}/lib \
        link=shared \
        variant=release \
        cxxflags=-w \
        install \
        && \
    rm -rf ${SCRATCH_DIR}

# Install Silo
ENV SILO_DIR=/opt/silo
RUN SILO_VERSION=4.10.2 && \
    SILO_URL=https://wci.llnl.gov/sites/wci/files/2021-01/silo-${SILO_VERSION}-bsd-smalltest.tgz && \
    SILO_ARCHIVE=silo.tar.gz && \
    wget --quiet ${SILO_URL} --output-document=${SILO_ARCHIVE} && \
    mkdir silo && \
    tar -xf ${SILO_ARCHIVE} -C silo --strip-components=1 && \
    cd silo && \
    ./configure --prefix=${SILO_DIR} --enable-fortran=no --prefix=${SILO_DIR} && \
    make -j${NPROCS} install && \
    cd ../../ && rm -r silo

# Install Kokkos
ARG KOKKOS_VERSION=3.4.01
ENV KOKKOS_DIR=/opt/kokkos
RUN KOKKOS_URL=https://github.com/kokkos/kokkos/archive/${KOKKOS_VERSION}.tar.gz && \
    KOKKOS_ARCHIVE=kokkos-${KOKKOS_VERSION}.tar.gz && \
    SCRATCH_DIR=/scratch && mkdir -p ${SCRATCH_DIR} && cd ${SCRATCH_DIR} && \
    wget --quiet ${KOKKOS_URL} --output-document=${KOKKOS_ARCHIVE} && \
    mkdir -p kokkos && \
    tar -xf ${KOKKOS_ARCHIVE} -C kokkos --strip-components=1 && \
    cd kokkos && \
    mkdir -p build && cd build && \
    cmake \
      -D CMAKE_INSTALL_PREFIX=${KOKKOS_DIR} \
      -D CMAKE_BUILD_TYPE=Debug \
      -D CMAKE_CXX_COMPILER=${SCRATCH_DIR}/kokkos/bin/nvcc_wrapper \
      -D Kokkos_ENABLE_SERIAL=ON \
      -D Kokkos_ENABLE_CUDA=ON \
      -D Kokkos_ENABLE_CUDA_LAMBDA=ON \
      -D Kokkos_ARCH_VOLTA70=ON \
    .. && \
    make -j${NPROCS} install && \
    rm -rf ${SCRATCH_DIR}

# Install ArborX
ENV ARBORX_DIR=/opt/arborx
RUN ARBORX_VERSION=v1.0 && \
    ARBORX_URL=https://github.com/arborx/ArborX/archive/${ARBORX_VERSION}.tar.gz && \
    ARBORX_ARCHIVE=arborx.tar.gz && \
    wget --quiet ${ARBORX_URL} --output-document=${ARBORX_ARCHIVE} && \
    mkdir arborx && \
    tar -xf ${ARBORX_ARCHIVE} -C arborx --strip-components=1 && \
    cd arborx && \
    mkdir -p build && cd build && \
    cmake \
      -D CMAKE_INSTALL_PREFIX=${ARBORX_DIR} \
      -D CMAKE_BUILD_TYPE=Debug \
      -D CMAKE_CXX_COMPILER=${KOKKOS_DIR}/bin/nvcc_wrapper \
      -D CMAKE_CXX_EXTENSIONS=OFF \
      -D CMAKE_PREFIX_PATH=${KOKKOS_DIR} \
    .. && \
    make -j${NPROCS} install && \
    cd ../.. && rm -r arborx

# Install HYPRE
ENV HYPRE_DIR=/opt/hypre
RUN HYPRE_VERSION=v2.22.1 && \
    HYPRE_URL=https://github.com/hypre-space/hypre/archive/${HYPRE_VERSION}.tar.gz && \
    HYPRE_ARCHIVE=hypre.tar.gz && \
    wget --quiet ${HYPRE_URL} --output-document=${HYPRE_ARCHIVE} && \
    mkdir hypre && \
    tar -xf ${HYPRE_ARCHIVE} -C hypre --strip-components=1 && \
    cd hypre && \
    mkdir -p build && cd build && \
    cmake \
      -D CMAKE_INSTALL_PREFIX=${HYPRE_DIR} \
      -D CMAKE_BUILD_TYPE=Debug \
      -D HYPRE_WITH_CUDA=ON \
      -D HYPRE_WITH_MPI=ON \
    ../src && \
    make -j${NPROCS} install && \
    cd ../.. && rm -r hypre

# Install Cabana (this version is August 2021)
ENV CABANA_DIR=/opt/cabana
RUN CABANA_VERSION=e6b5c5b5b28d7eb13ef8cbaee5d68e9cd541f37e && \
    CABANA_URL=https://github.com/ECP-copa/Cabana/archive/${CABANA_VERSION}.tar.gz && \
    CABANA_ARCHIVE=cabana.tar.gz && \
    wget --quiet ${CABANA_URL} --output-document=${CABANA_ARCHIVE} && \
    mkdir cabana && \
    tar -xf ${CABANA_ARCHIVE} -C cabana --strip-components=1 && \
    cd cabana && \
    mkdir -p build && cd build && \
    cmake \
      -D CMAKE_INSTALL_PREFIX=${CABANA_DIR} \
      -D CMAKE_BUILD_TYPE=Debug \
      -D CMAKE_CXX_COMPILER=$KOKKOS_DIR/bin/nvcc_wrapper \
      -D CMAKE_PREFIX_PATH="$KOKKOS_DIR;$ARBORX_DIR;$HYPRE_DIR" \
      -D MPIEXEC_MAX_NUMPROCS=1 \
      .. && \
    make -j${NPROCS} install && \
    cd ../.. && rm -r cabana

FROM image-cloud.luchentech.com/hpcaitech/pytorch-cuda:2.3.0-12.1.0


ENV PIP_REQUIRE_HASHES=

ARG http_proxy
ARG https_proxy
ENV http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    HTTP_PROXY=${http_proxy} \
    HTTPS_PROXY=${https_proxy}

RUN apt-get update && \
    apt-get install -y gcc-9 g++-9 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100 \
                        --slave   /usr/bin/g++ g++ /usr/bin/g++-9 && \
    rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

RUN conda update -n base -c defaults -y conda openssl ca-certificates && \
    conda config --remove-key channels || true && \
    conda config --add channels conda-forge && \
    conda config --add channels bioconda && \
    conda config --add channels defaults && \
    conda config --set channel_priority strict && \
    conda config --set show_channel_urls yes

RUN conda install -n pytorch -y -c conda-forge -c bioconda \
        openmm=8.2.0 \
        pdbfixer \
        hmmer=3.3.2 \
        hhsuite=3.3.0 \
        scipy==1.11.4 \
        kalign2=2.04 && \
    conda clean -afy

RUN mkdir -p /etc/pip && \
    printf "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\ntimeout = 120\nretries = 4\n" > /etc/pip.conf

RUN /opt/conda/envs/pytorch/bin/pip install --no-cache-dir \
        biopython==1.79 \
        dm-tree==0.1.6 \
        ml-collections==0.1.0 \
        ray==2.47.1 \
        pyarrow \
        pandas \
        einops \
        colossalai==0.5.0


RUN git clone --depth 1 https://github.com/hpcaitech/FastFold.git && \
    cd FastFold && \
    sed -i 's/-std=c++14/-std=c++17/g' setup.py && \
    grep -RIl -- '-std=c++14' . | xargs -r sed -i 's/-std=c++14/-std=c++17/g' && \
    grep -RIl -- 'np\.object_' fastfold | xargs -r sed -i 's/np\.object_/object/g' && \
    grep -RIl -- 'np\.object' fastfold | xargs -r sed -i 's/np\.object/object/g' && \
    grep -RIl -- 'np\.int\>'  fastfold | xargs -r sed -i 's/np\.int\>/int/g' && \
    grep -REIl "from[[:space:]]+colossalai\.context(\.parallel_mode)?[[:space:]]+import[[:space:]]+ParallelMode" fastfold \
      | xargs -r sed -i -E 's#from[[:space:]]+colossalai\.context(\.parallel_mode)?[[:space:]]+import[[:space:]]+ParallelMode#from colossalai.legacy.context import ParallelMode#g'
    grep -RIl "from colossalai\.core import global_context as gpc" fastfold \
      | xargs -r sed -i 's/from colossalai\.core import global_context as gpc/from colossalai.legacy.core import global_context as gpc/g' && \
    grep -RIl "from colossalai\.context(\.parallel_mode)? import ParallelMode" fastfold \
      | xargs -r sed -i -E 's/from colossalai\.context(\.parallel_mode)? import ParallelMode/from colossalai.legacy.context import ParallelMode/g' && \
    sed -i '1i from colossalai.legacy import launch_from_torch as _ff_launch_from_torch' fastfold/distributed/core.py && \
    sed -i 's/colossalai\.legacy\.launch\.launch_from_torch(/_ff_launch_from_torch(/g' fastfold/distributed/core.py && \
    sed -i 's/colossalai\.launch_from_torch(/_ff_launch_from_torch(/g' fastfold/distributed/core.py && \
    /opt/conda/envs/pytorch/bin/python setup.py install && \
    cd / && rm -rf FastFold

WORKDIR /workspace
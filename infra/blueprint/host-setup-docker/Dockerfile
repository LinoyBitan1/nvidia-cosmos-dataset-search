# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

FROM nvcr.io/nvidia/ai-workbench/python-cuda122:1.0.3

RUN apt-get update && apt-get install -y unzip gettext less

WORKDIR /workspace
RUN mkdir packages
WORKDIR /workspace/packages

# aws-cli v2.17.13
RUN curl -LO "https://awscli.amazonaws.com/awscli-exe-linux-x86_64-2.17.13.zip"
RUN unzip awscli-exe-linux-x86_64-2.17.13.zip
RUN ./aws/install

# eksctl latest (using recommended download method)
RUN curl -sLO "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_Linux_amd64.tar.gz"
RUN tar -xzf eksctl_Linux_amd64.tar.gz
RUN mv eksctl /usr/local/bin

# kubectl v1.30.1
RUN curl -LO "https://dl.k8s.io/release/v1.30.1/bin/linux/amd64/kubectl"
RUN install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# helm v3.17.0
RUN curl -LO "https://get.helm.sh/helm-v3.17.0-linux-amd64.tar.gz"
RUN tar -xzf helm-v3.17.0-linux-amd64.tar.gz
RUN mv linux-amd64/helm /usr/local/bin/helm

# Python v3.10 and pip v22.02 is already installed

# s5cmd v2.3.0
RUN curl -LO "https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_linux_amd64.deb"
RUN apt install ./s5cmd_2.3.0_linux_amd64.deb

# kaggle v1.6.17
RUN pip install kaggle==1.6.17

# python magic v0.4.27
RUN apt-get install -y libmagic-dev
RUN pip install python-magic==0.4.27

# Update/remove some packages to avoid security vulnerabilities
RUN pip uninstall -y jupyterlab jupyter_server jupyter-lsp
RUN pip install tornado==6.4.2

# Generate secret encryption ley
RUN pip install cryptography==42.0.8

# Change directory and delete downloaded temporary files
WORKDIR /workspace
RUN rm -rf /workspace/packages

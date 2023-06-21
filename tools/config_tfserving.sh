#!/usr/bin/env bash

# Config the serving for build serving with TFRA OPs.
# Usage:
#   Param $1 is the branch name of TFRA
#   Param $2 is the serving root directory.
#   Param $3 is the flag indicating if enable CUDA for GPU.
#   ```shell
#     cd /recommenders_addons/tools
#     bash ./config_tfserving.sh "r0.6" /root/of/serving/ "1"
#     cd /root/of/serving/
#     ./tools/run_in_docker.sh bazel build tensorflow_serving/model_servers:tensorflow_model_server
#   ```

tfra_branch=$1
tfserving_root=$2
is_gpu=$3

# Pleas modify this mapping when update TFRA.
declare -A serving_version
serving_version["master"]="2.8.3"

# 1. copy directory to tfserving root
cp -r ../tensorflow_recommenders_addons ${tfserving_root}/
cp -r ../build_deps ${tfserving_root}/

# 2. Padding the WORKSPACE
cat "../WORKSPACE"| tail -n +2 >> ${tfserving_root}/WORKSPACE

# 3. Padding the tools/run_in_docker.sh
file="${tfserving_root}/tools/run_in_docker.sh"
original_docker="tensorflow\/serving:nightly-devel"
replacement_docker="tfra\/serving:${serving_version[$tfra_branch]}-devel"
if [[ "$is_gpu" == "1" ]]; then
  replacement_docker="tfra\/serving:${serving_version[$tfra_branch]}-devel-gpu"
fi

sed -i "s/$original_docker/$replacement_docker/g" $file

# 4. Padding .bazelrc
file=${tfserving_root}/.bazelrc
cat "serving_padding/.bazelrc_padding" >> $file

if [[ "$is_gpu" == "1" ]]; then
  cat "serving_padding/.bazelrc_gpu_padding" >> $file
fi

# 5. Padding tensorflow_serving/model_servers/BUILD
file=${tfserving_root}/tensorflow_serving/model_servers/BUILD
target_string="org_tensorflow_text\/\/tensorflow_text:ops_lib"

sed -i "/$target_string/a \\
    \"\/\/tensorflow_recommenders_addons\/dynamic_embedding\/core:_cuckoo_hashtable_ops.so\", \\
    \"\/\/tensorflow_recommenders_addons\/dynamic_embedding\/core:_math_ops.so\", \\
    \"\/\/tensorflow_recommenders_addons\/dynamic_embedding\/core:_redis_table_ops.so\",
    " $file

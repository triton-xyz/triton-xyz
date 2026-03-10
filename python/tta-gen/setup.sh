pushd third_party
git clone https://gitcode.com/Ascend/triton-ascend.git
pushd triton-ascend
git checkout -b test 115f51f71917e6836c3138a8f0d52fb71caf1d63
popd
popd

rm -f $PWD/third_party/triton-ascend/third_party/ascend/unittest/generalization_cases/conftest.py
ln -s $PWD/python/tta-gen/conftest.py \
  $PWD/third_party/triton-ascend/third_party/ascend/unittest/generalization_cases/conftest.py

rm -f $PWD/third_party/triton-ascend/third_party/ascend/unittest/generalization_cases/torch_npu.py
ln -s $PWD/python/tta-gen/torch_npu.py \
  $PWD/third_party/triton-ascend/third_party/ascend/unittest/generalization_cases/torch_npu.py

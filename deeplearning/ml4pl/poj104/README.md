
# Install pytorch and pytorch-geometric
```
pip install torch
# set paths: 

pip install --verbose --no-cache-dir torch-scatter
pip install --verbose --no-cache-dir torch-sparse
pip install --verbose --no-cache-dir torch-cluster
pip install --verbose --no-cache-dir torch-spline-conv (optional)
pip install torch-geometric
```


# How to run the dataset generation?

* Install pytorch-geometric (see their website)

* `cd .../ProGraML`
* `bazel test //deeplearning/ml4pl/graphs/unlabelled/llvm2graph/...`
*  Some of those tests should fail because `llvm_mac` or `llvm_linux` doesn't exist.
* Find and copy the above folder to `ProGraML/llvm_X`. It can be found in the build files `ProGraML/bazel-bin/......../llvm_X`.
* `cd ProGraML/deeplearning/ml4pl/poj104`
* `jupyter notebook`
* Run the `generate_programl_dataset` notebook and set your `ds_basepath` to the folder where you want the dataset folder to appear.

# Maybe (on AULT)
```
export LD_LIBRARY_PATH=/users/zfisches/miniconda3/lib/:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
```

See https://stackoverflow.com/questions/49875588/importerror-lib64-libstdc-so-6-version-cxxabi-1-3-9-not-found
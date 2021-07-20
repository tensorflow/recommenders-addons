#### C++
C++ code should conform to [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

Addons uses [clang-format](https://clang.llvm.org/docs/ClangFormat.html)
to check your C/C++ changes. Sometimes you have some manually formatted
code that you don’t want clang-format to touch.
You can disable formatting like this:

```cpp
int formatted_code;
// clang-format off
    void    unformatted_code  ;
// clang-format on
void formatted_code_again;
```

Install Clang-format 9 for Ubuntu:

```bash
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add - 
sudo add-apt-repository -u 'http://apt.llvm.org/bionic/ llvm-toolchain-bionic-9 main'
sudo apt install clang-format-9
```

format all with:
```bash
clang-format-9 -i --style=google ./tensorflow_recommenders_addons/**/*.cc ./tensorflow_recommenders_addons/**/*.h
```

Install Clang-format for MacOS:
```bash
brew update
brew install clang-format
```

format all with:
```bash
clang-format -i --style=google **/*.cc tensorflow_recommenders_addons/**/*.h
```

#### Python
Recommenders Addons use [Yapf](https://github.com/google/yapf) to format our code.
The continuous integration check will fail if you do not use it.

Install them with:
```
pip install yapf
```

Be sure to run it before you push your commits, otherwise the CI will fail!

```
yapf --style=./.yapf -ir ./**/*.py
```

#### Bazel BUILD
Use [buildifier](https://github.com/bazelbuild/buildtools/blob/master/buildifier/README.md) in project [bazelbuild/buildtools](https://github.com/bazelbuild/buildtools) to format the bazel code.

Install it with:
```bash
git clone https://github.com/bazelbuild/buildtools.git
cd buildtools
bazel build //buildifier
```
Then copy the binary to directory on $PATH. (such as "/usr/local/bin")
```bash
cp bazel-bin/buildifier/buildifier_/buildifier /usr/local/bin
```

Run following commmand to see whether if installation ok:
```bash
buildifier --version
```

Use `buildifier`
```bash
buildifier -mode diff ${your_file_name}
```
to see formating problem in the BUILD file, or:
```bash
buildifier -mode diff ${directory}
```
for all BUILD files in ${directory}.

#### TensorFlow Conventions

Follow the guidance in the [TensorFlow Style Guide - Conventions](https://www.tensorflow.org/community/contribute/code_style#tensorflow_conventions_and_special_uses).

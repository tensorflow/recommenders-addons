FROM python:3.6

COPY tools/install_deps /install_deps
RUN pip install -r /install_deps/yapf.txt

COPY tools/install_deps/buildifier.sh ./buildifier.sh
RUN bash buildifier.sh

COPY tools/install_deps/clang-format.sh ./clang-format.sh
RUN bash clang-format.sh

WORKDIR /recommenders-addons


CMD ["python", "tools/pre_commit_format.py"]

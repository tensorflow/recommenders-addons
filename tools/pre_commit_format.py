#!/usr/bin/env python
from subprocess import check_call, CalledProcessError


def check_bash_call(string):
  check_call(["bash", "-c", string])


def _run_format():
  files_changed = False

  try:
    check_bash_call(
        "find . -name '*.py' -print0 | xargs -0 yapf --style=./.yapf -dr")
  except CalledProcessError:
    check_bash_call(
        "find . -name '*.py' -print0 | xargs -0 yapf --style=./.yapf -ir")
    files_changed = True

  try:
    check_bash_call("buildifier -mode=check -r .")
  except CalledProcessError:
    check_bash_call("buildifier -r .")
    files_changed = True

  # todo: find a way to check if files changed
  # see https://github.com/DoozyX/clang-format-lint-action for inspiration
  check_bash_call(
      "shopt -s globstar && clang-format-9 -i --style=google **/*.cc **/*.h",)

  if files_changed:
    print("Some files have changed.")
    print("Please do git add and git commit again")
  else:
    print("No formatting needed.")

  if files_changed:
    exit(1)


def run_format():
  try:
    _run_format()
  except CalledProcessError as error:
    print("Pre-commit returned exit code", error.returncode)
    exit(error.returncode)


if __name__ == "__main__":
  run_format()

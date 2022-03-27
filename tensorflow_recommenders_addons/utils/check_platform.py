# Copyright 2022 The TensorFlow Recommenders-Addons Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Check the system and architecture types
"""
import platform


def is_macos():
  return platform.system() == "Darwin"


def is_windows():
  return platform.system() == "Windows"


def is_linux():
  return platform.system() == "Linux"


def is_arm64():
  return platform.machine() == "arm64"


def is_raspi_arm():
  return os.uname()[4] == "armv7l"

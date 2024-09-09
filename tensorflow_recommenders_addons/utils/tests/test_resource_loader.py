import unittest
from unittest.mock import patch, Mock

import pkg_resources
import tensorflow as tf

from tensorflow_recommenders_addons.utils.resource_loader import abi_is_compatible


class TestTensorFlowCompatibility(unittest.TestCase):

  @patch('pkg_resources.get_distribution')
  @patch('tensorflow.__version__', '2.12.0')
  def test_compatible_version(self, mock_get_distribution):
    mock_pkg = Mock()
    mock_requirement = Mock()
    mock_requirement.name = 'tensorflow'
    mock_requirement.specs = [('>=', '2.11.0'), ('<=', '2.16.2')]
    mock_pkg.requires.return_value = [mock_requirement]
    mock_get_distribution.return_value = mock_pkg
    self.assertTrue(abi_is_compatible())

    @patch('pkg_resources.get_distribution')
    @patch('tensorflow.__version__', '2.10.0')
    def test_incompatible_version_below_range(self, mock_get_distribution):
      mock_pkg = Mock()
      mock_requirement = Mock()
      mock_requirement.name = 'tensorflow'
      mock_requirement.specs = [('>=', '2.11.0'), ('<=', '2.16.2')]
      mock_pkg.requires.return_value = [mock_requirement]
      mock_get_distribution.return_value = mock_pkg
      self.assertFalse(abi_is_compatible())

    @patch('pkg_resources.get_distribution')
    @patch('tensorflow.__version__', '2.16.0')
    def test_incompatible_version_above_range(self, mock_get_distribution):
      mock_pkg = Mock()
      mock_requirement = Mock()
      mock_requirement.name = 'tensorflow'
      mock_requirement.specs = [('>=', '2.11.0'), ('<=', '2.16.2')]
      mock_pkg.requires.return_value = [mock_requirement]
      mock_get_distribution.return_value = mock_pkg
      self.assertFalse(abi_is_compatible())

    @patch('pkg_resources.get_distribution')
    @patch('tensorflow.__version__', '2.13.0-dev20240528')
    def test_dev_version(self, mock_get_distribution):
      mock_pkg = Mock()
      mock_requirement = Mock()
      mock_requirement.name = 'tensorflow'
      mock_requirement.specs = [('>=', '2.11.0'), ('<=', '2.16.2')]
      mock_pkg.requires.return_value = [mock_requirement]
      mock_get_distribution.return_value = mock_pkg
      self.assertFalse(abi_is_compatible())

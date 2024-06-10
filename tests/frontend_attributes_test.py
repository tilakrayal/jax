# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests whether the frontend attributes added by the context manager are
correctly propagated to the jaxpr and mlir.
"""
from collections.abc import Sequence
from absl import app
import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
import jax.scipy as jsp
from jax._src import test_util as jtu
from absl.testing import absltest
from absl.logging import logging
from jax._src import core
from typing import cast
from jax._src import dispatch
from jax._src import core
from jax._src.lax import lax

class FrontendAttributesTest(absltest.TestCase):

  def test_no_attributes(self):
    @jax.jit
    def f(a, b):
      return a + b
    f_lowered_text = f.lower(1., 2.).as_text()
    self.assertIn("mhlo.frontend_attributes = {}", f_lowered_text)

  def test_f_jitted_jaxpr(self):
    @jax.jit
    def f(a, b):
      with jax.attributes(a="b"):
        return a + b
    f_jaxpr = jax.make_jaxpr(f)(1, 2)
    eqns = f_jaxpr.eqns
    self.assertIn("# [{'a': 'b'}]", str(eqns[0]))

  def test_f_jitted_mlir(self):
    @jax.jit
    def f(a, b):
      with jax.attributes(a="b"):
        return a + b
    f_lowered_text = f.lower(1., 2.).as_text()
    self.assertIn("mhlo.frontend_attributes = {a = \"b\"}", f_lowered_text)

  def test_f_nonjitted_mlir(self):
    def f_add(a, b):
      return dispatch.apply_primitive(lax.add_p, a, b)
    arg1 = jax.numpy.arange(2)
    with jax.attributes(a="b"):
      self.assertIn("mhlo.frontend_attributes = {a = \"b\"}", jax.jit(f_add).lower(arg1, arg1).as_text())

  def test_f_attributes_scope(self):
    with jax.attributes(a="b"):
      @jax.jit
      def f(a, b):
        return a + b
    # Expect no attributes
    f_lowered_text = f.lower(1., 2.).as_text()
    self.assertIn("mhlo.frontend_attributes = {}", f_lowered_text)

  def test_f_attributes_overwrite(self):
    with jax.attributes(a="b"):
      @jax.jit
      def f(a, b):
        with jax.attributes(a="c"):
          return a + b
      f_lowered_text = f.lower(1., 2.).as_text()
      self.assertIn("mhlo.frontend_attributes = {a = \"c\"}", f_lowered_text)

  def test_f_attributes_merge(self):
    with jax.attributes(key1="val1"):
      @jax.jit
      def f(a, b):
        with jax.attributes(key2="val2"):
          return a + b
      f_jaxpr = jax.make_jaxpr(f)(1., 2.)
      eqns = f_jaxpr.eqns
      print("merge: ", str(eqns[0]))
      f_lowered_text = f.lower(1., 2.).as_text()
      self.assertIn("mhlo.frontend_attributes = {key1 = \"val1\", key2 = \"val2\"}", f_lowered_text)

  def test_attr_caching_jit_mlir(self):
    @jax.jit
    def f_add_jit(a, b):
      return a + b
    with jax.attributes(b="c"):
      f_add_lowered1 = f_add_jit.lower(2., 3.).as_text()
    # Expect no attributes in the mlir.
    f_add_lowered2 = f_add_jit.lower(1., 2.).as_text()
    with jax.attributes(c="d"):
      f_add_lowered3 = f_add_jit.lower(4., 5.).as_text()
    self.assertIn("mhlo.frontend_attributes = {b = \"c\"}", f_add_lowered1)
    self.assertIn("mhlo.frontend_attributes = {}", f_add_lowered2)
    self.assertIn("mhlo.frontend_attributes = {c = \"d\"}", f_add_lowered3)

  def test_attr_caching_nonjit_mlir(self):
    def f_add(a, b):
      return dispatch.apply_primitive(lax.add_p, a, b)
    arg1 = jax.numpy.arange(2)
    arg2 = jax.numpy.arange(2) + 1
    arg3 = jax.numpy.arange(2) + 2
    with jax.attributes(b="c"):
      self.assertIn("mhlo.frontend_attributes = {b = \"c\"}", jax.jit(f_add).lower(arg1, arg1).as_text())
    # Expect no attributes in the jaxpr.
    self.assertIn("mhlo.frontend_attributes = {}", jax.jit(f_add).lower(arg2, arg2).as_text())
    print(f_add(arg2, arg2))
    with jax.attributes(c="d"):
      self.assertIn("mhlo.frontend_attributes = {c = \"d\"}", jax.jit(f_add).lower(arg3, arg3).as_text())
  
  def test_axpy(self):
    @jax.jit
    def axpy(a, x, y):
      with jax.attributes(a="b"):
        return a * x + y
    self.assertIn("mhlo.frontend_attributes = {a = \"b\"}", axpy.lower(1., 2., 3.).as_text())
  
  def test_while(self):
    @jax.jit
    def f(a):
      with jax.attributes(a="b"):
        return jax.lax.while_loop(lambda x: x < 10, lambda x: x + 1, a)
    self.assertIn("mhlo.frontend_attributes = {a = \"b\"}", f.lower(1.).as_text())
  
  # def test_jax_inv(self):
  #   @jax.jit
  #   def compute_inv(a):
  #     matrix = 
  #     data_inv_cov = jnp.linalg.inv(data_covariance)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())

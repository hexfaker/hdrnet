# Copyright 2016 Google Inc.
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
"""TF graph utilities."""

import tensorflow as tf
import yaml


def get_model_params(sess, param_collection="model_params"):
    pcoll = tf.get_collection(param_collection)
    params_ = {p.name.split(':')[0]: p for p in pcoll}
    model_params = sess.run(params_)
    return model_params


def get_config(path, name):
    with open(path, 'r') as f:
        conf = yaml.load(f)

    if name in conf:
        return conf[name]

    return {}


def dump_config(path, model, data):
    with open(path, 'w') as f:
        yaml.dump(dict(model=model, data=data), f)


def load_stats(path):
    with open(path, 'r') as f:
        stats = yaml.load(f)

    if stats is None:
        return {}
    return stats

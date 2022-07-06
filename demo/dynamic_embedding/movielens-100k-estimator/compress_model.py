import sys, os, shutil, argparse
import tensorflow.compat.v1 as tf
import tensorflow_recommenders_addons as tfra
from tensorflow.python.saved_model import loader_impl
from tensorflow.core.framework import node_def_pb2, attr_value_pb2

OP_PREFIX = 'TFRA>CuckooHashTable'


def model_info(model_dir):
  print('Model dir:\t{}'.format(model_dir))

  pb_size = os.path.getsize(os.path.join(model_dir, 'saved_model.pb'))

  variable_path = os.path.join(model_dir, 'variables')
  variables_size = 0
  for f in os.listdir(variable_path):
    variables_size += os.path.getsize(os.path.join(variable_path, f))

  print('Graph size:\t{:.2f} MiB'.format(pb_size / 2**20))
  print('Variables size:\t{:.2f} MiB'.format(variables_size / 2**20))
  print('Total size:\t{:.2f} MiB'.format((pb_size + variables_size) / 2**20))


def change_model(model_dir, dst_dir):

  def set_attr_dtype(node, key, value):
    try:
      node.attr[key].CopyFrom(
          attr_value_pb2.AttrValue(type=value.as_datatype_enum))
    except KeyError:
      pass

  def create_cast_node(name, srcT, desT, inputs):
    node = node_def_pb2.NodeDef()
    node.op = 'Cast'
    node.name = name
    for input_name in inputs:
      node.input.extend([input_name])
    set_attr_dtype(node, 'SrcT', srcT)
    set_attr_dtype(node, 'DstT', desT)
    return node

  saved_model = loader_impl.parse_saved_model(model_dir)
  nodes = saved_model.meta_graphs[0].graph_def.node

  map_node_name_to_node = {node.name: node for node in nodes}

  cuckoo_import_nodes = [n for n in nodes if n.op == OP_PREFIX + 'Import']
  cuckoo_export_nodes = [n for n in nodes if n.op == OP_PREFIX + 'Export']
  cuckoo_table_nodes = [n for n in nodes if n.op == OP_PREFIX + 'OfTensors']
  cuckoo_find_nodes = [
      n for n in nodes
      if n.op == OP_PREFIX + 'Find' or n.op == OP_PREFIX + 'FindWithExists'
  ]

  #find all hash table value variable names and
  #change the dtype in CuckooHashTableImport Op and RestoreV2 Op
  ht_values = []
  for import_node in cuckoo_import_nodes:
    restore_node = map_node_name_to_node[import_node.input[2].split(':')[0]]
    restore_idx = int(import_node.input[2].split(':')[1])
    assert (restore_node.op == 'RestoreV2')
    assert (restore_node.attr['dtypes'].list.type[restore_idx] ==
            tf.float32.as_datatype_enum)

    tensor_names_node = map_node_name_to_node[restore_node.input[1].split(':')
                                              [0]]
    ht_value = tensor_names_node.attr['value'].tensor.string_val[
        restore_idx].decode()
    ht_values.append(ht_value)

    restore_node.attr['dtypes'].list.type[
        restore_idx] = tf.float16.as_datatype_enum
    import_node.attr['Tout'].type = tf.float16.as_datatype_enum

  print('Found {} dynamic embedding hash table value variables:\n{}'.format(
      len(ht_values), ht_values))

  #change the dtype in CuckooHashTableExport Op and SaveV2 op
  for n in cuckoo_export_nodes:
    n.attr['Tvalues'].type = tf.float16.as_datatype_enum

  for n in nodes:
    if n.op == 'SaveV2':
      for idx, input in enumerate(n.input[3:]):
        node = map_node_name_to_node[input.split(':')[0]]

        if node in cuckoo_export_nodes:
          if n.attr['dtypes'].list.type[idx] == tf.float32.as_datatype_enum:
            n.attr['dtypes'].list.type[idx] = tf.float16.as_datatype_enum

  #change the dtype in CuckooHashTableOfTensors Op
  for n in cuckoo_table_nodes:
    n.attr['value_dtype'].type = tf.float16.as_datatype_enum

  #change the dtype in CuckooHashTableFind Op
  #and add cast node before default value input
  for n in cuckoo_find_nodes:
    n.attr['Tout'].type = tf.float16.as_datatype_enum

    default_value = n.input[2]
    node_name = n.name + '/default_value_cast'
    cast_node = create_cast_node(node_name, tf.float32, tf.float16,
                                 [default_value])
    nodes.extend([cast_node])
    n.input[2] = node_name

  #add cast node after CuckooHashTableFind Op
  for n in nodes:
    for idx, input in enumerate(n.input):
      if input in [n.name for n in cuckoo_find_nodes]:
        n.input[idx] = input + '/Cast'
  for n in cuckoo_find_nodes:
    node_name = n.name + '/Cast'
    cast_node = create_cast_node(node_name, tf.float16, tf.float32, [n.name])
    nodes.extend([cast_node])

  if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)
  os.mkdir(dst_dir)
  new_saved_model = os.path.join(dst_dir, 'saved_model.pb')
  with open(new_saved_model, 'wb') as f:
    f.write(saved_model.SerializeToString())
  print('Compressed model saved to {}'.format(new_saved_model))
  return ht_values


def compress_variables(model_dir, dst_dir, variables):
  dst_model_variables_dir = os.path.join(dst_dir, 'variables')
  os.mkdir(dst_model_variables_dir)

  model_variables_dir = os.path.join(model_dir, 'variables')
  checkpoint_file = os.path.join(model_variables_dir, 'checkpoint')
  with open(checkpoint_file, 'w') as f:
    f.write('model_checkpoint_path: "variables"\n')

  #checkpoint_stat = tf.train.get_checkpoint_state(model_variables_dir)
  checkpoint_dict = tf.train.list_variables(model_variables_dir)

  with tf.Session() as sess:
    for var_name, _ in checkpoint_dict:
      #print('variables in checkpoint {} {}'.format(var_name, _))
      np_var = tf.train.load_variable(model_variables_dir, var_name)

      if (var_name in variables):
        np_var = np_var.astype(tf.float16.as_numpy_dtype)
        new_var = tf.Variable(np_var, name=var_name, dtype=tf.float16)
      else:
        new_var = tf.Variable(np_var, name=var_name)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.save(sess,
               os.path.join(dst_model_variables_dir, 'variables'),
               write_meta_graph=False,
               write_state=False)

  print('Variables in {} compressed to {}'.format(model_variables_dir,
                                                  dst_model_variables_dir))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description=
      'Compress TFRA dynamic embedding table values from float32 to float16.')
  parser.add_argument('--model_dir',
                      type=str,
                      required=True,
                      help='source SavedModel model dir')
  parser.add_argument(
      '--dst_dir',
      type=str,
      default='compressed_model',
      help='destination dir of compressed model, default: ./compressed_model')
  args = parser.parse_args()

  print('\n-------- Original SavedModel infomation --------\n')
  model_info(args.model_dir)

  print('\n-------- Compress variables and change model --------\n')
  value_variables = change_model(args.model_dir, args.dst_dir)
  compress_variables(args.model_dir, args.dst_dir, value_variables)

  print('\n-------- Compressed SavedModel infomation --------\n')
  model_info(args.dst_dir)

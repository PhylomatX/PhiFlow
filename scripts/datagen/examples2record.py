import os
import json
import pickle as pkl
import tensorflow as tf

from absl import app
from absl import flags

flags.DEFINE_string("ex_path", None, help="Path to examples")
flags.DEFINE_string("save_path", None, help="Path where record should get saved.")
flags.DEFINE_string("split", "train", help="Type of dataset")
FLAGS = flags.FLAGS


def main(_):
    files = os.listdir(FLAGS.ex_path)

    examples = []
    vel_wf = None
    acc_wf = None
    particle_num_wf = None
    metadata = None

    for file in files:
        with open(os.path.join(FLAGS.ex_path, file), 'rb') as f:
            data = pkl.load(f)

        examples = examples + data['examples']
        if vel_wf is None:
            vel_wf = data['vel_wf']
        else:
            vel_wf.merge(data['vel_wf'])
        if acc_wf is None:
            acc_wf = data['acc_wf']
        else:
            acc_wf.merge(data['acc_wf'])
        if particle_num_wf is None:
            particle_num_wf = data['particle_num_wf']
        else:
            particle_num_wf.merge(data['particle_num_wf'])
        data.pop('examples')
        data.pop('vel_wf')
        data.pop('acc_wf')
        data.pop('particle_num_wf')
        if metadata is None:
            metadata = data
        else:
            for key in metadata:
                if key == 'dataset_size':
                    metadata[key] += data[key]
                else:
                    assert metadata[key] == data[key], "Examples are from datasets which are not compatible!"

    with tf.io.TFRecordWriter(os.path.join(FLAGS.save_path, f'{FLAGS.split}.tfrecord')) as writer:
        for example in examples:
            writer.write(example.SerializeToString())

    metadata['vel_mean'] = vel_wf.mean.tolist()
    metadata['acc_mean'] = acc_wf.mean.tolist()
    metadata['particle_num_mean'] = float(particle_num_wf.mean)
    metadata['vel_std'] = vel_wf.std.tolist()
    metadata['acc_std'] = acc_wf.std.tolist()
    metadata['particle_num_std'] = float(particle_num_wf.std)

    with open(os.path.join(FLAGS.save_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)


if __name__ == '__main__':
    app.run(main)

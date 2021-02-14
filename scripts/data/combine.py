import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import json
import pickle as pkl
import tensorflow as tf

from absl import app
from absl import flags

flags.DEFINE_string("path", None, help="Path to examples")
flags.DEFINE_string("save_path", None, help="Path where record should get saved.")
flags.DEFINE_string("split", "train", help="Type of dataset")
flags.DEFINE_bool("ignore_scene", False, help="Allow combining sets with different scene params")
FLAGS = flags.FLAGS


def main(_):
    files = glob.glob(os.path.join(FLAGS.path, '*.pkl'))

    examples = []
    vel_wf = None
    acc_wf = None
    particle_num_wf = None
    radius_stats = []
    particle_num_range = None
    metadata = None

    for file in files:
        with open(file, 'rb') as f:
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
        if len(radius_stats) == 0:
            for ix in range(len(data['radius_stats'])):
                radius_stats.append(data['radius_stats'][ix])
        else:
            for ix in range(len(data['radius_stats'])):
                radius_stats[ix] += data['radius_stats'][ix]
        if particle_num_range is None:
            particle_num_range = list(data['particle_num_range'])
        else:
            curr_num_range = data['particle_num_range']
            if curr_num_range[0] < particle_num_range[0]:
                particle_num_range[0] = curr_num_range[0]
            if curr_num_range[1] > particle_num_range[1]:
                particle_num_range[1] = curr_num_range[1]

        data.pop('examples')
        data.pop('vel_wf')
        data.pop('acc_wf')
        data.pop('particle_num_wf')
        data.pop('radius_stats')
        data.pop('particle_num_range')

        if metadata is None:
            metadata = data
        else:
            for key in metadata:
                if key == 'dataset_size':
                    metadata[key] += data[key]
                else:
                    if key == 'scene' and FLAGS.ignore_scene:
                        continue
                    assert metadata[key] == data[key], "Examples are from datasets which are not compatible!"

    with tf.io.TFRecordWriter(os.path.join(FLAGS.save_path, f'{FLAGS.split}.tfrecord')) as writer:
        for example in examples:
            writer.write(example.SerializeToString())

    if vel_wf is not None:
        metadata['vel_mean'] = vel_wf.mean.tolist()
        metadata['vel_std'] = vel_wf.std.tolist()
    if acc_wf is not None:
        metadata['acc_mean'] = acc_wf.mean.tolist()
        metadata['acc_std'] = acc_wf.std.tolist()
    if particle_num_wf is not None:
        metadata['particle_num_mean'] = float(particle_num_wf.mean)
        metadata['particle_num_std'] = float(particle_num_wf.std)
    metadata['radius_stats'] = list(map(lambda x: list(x / len(files)), radius_stats))
    metadata['particle_num_range'] = particle_num_range

    with open(os.path.join(FLAGS.save_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)


if __name__ == '__main__':
    app.run(main)

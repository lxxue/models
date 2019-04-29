from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import numpy as np

import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt import show_and_tell_model
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
# tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
tf.flags.DEFINE_string("feats_file", "", "feats filename")

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    tf.logging.info("Building model")
    model = show_and_tell_model.ShowAndTellModel(configuration.ModelConfig(), mode="inference")
    model.build()
    saver = tf.train.Saver()
    # model = inference_wrapper.InferenceWrapper()
    # restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
    #                                            FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  # vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  filenames = []
  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  feats = np.zeros((len(filenames), 512), dtype=np.float32)
  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    tf.logging.info("Loading model from checkpoint:%s", FLAGS.checkpoint_path)
    saver.restore(sess, FLAGS.checkpoint_path)
    tf.logging.info("Successfully loaded checkpoint")
    # restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    #generator = caption_generator.CaptionGenerator(model, vocab)

    for i, filename in enumerate(filenames):
      with tf.gfile.GFile(filename, "rb") as f:
        image = f.read()
      feats[i] = sess.run(fetches=model.image_embeddings, feed_dict={"image_feed:0": image})
      if i % 100 == 99:
        tf.logging.info("%d / %d", i+1, len(filenames))
      # print(img_embedding)
      #captions = generator.beam_search(sess, image)
      #print("Captions for image %s:" % os.path.basename(filename))
      #for i, caption in enumerate(captions):
        # Ignore begin and end words.
      #  sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
      #  sentence = " ".join(sentence)
      #  print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
  np.save(FLAGS.feats_file, feats)

if __name__ == "__main__":
  tf.app.run()

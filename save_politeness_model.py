
"""
  Saves the model in a format suitable for serving.
  The saved model is used by the poliserv.py application.
  It can also be served by the tensorflow_model_server.

"""

# %%
import tensorflow as tf
import tensorflow_hub as hub
import logging
# Suppress TF warnings.
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# %%
tf.keras.backend.set_learning_phase(0) # Ignore dropout at inference

# %%
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3" 

# %%
#Note: make sure to delete any previously saved model in export_dir.
export_dir = "./models/use/1"
with tf.Session() as session:
  embed = hub.Module(module_url)
  text_input = tf.placeholder(dtype=tf.string, shape=[None])  
  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())
  model = tf.keras.models.load_model('./model.h5')
  model.summary()
  tf.saved_model.simple_save(session,
        export_dir,
        inputs={'text': model.input},
        outputs={t.name:t for t in model.outputs},
        legacy_init_op=tf.tables_initializer())

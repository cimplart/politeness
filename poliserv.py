"""
 A simple model serving application.

"""

import os
from flask import Flask, request
from flask import make_response
from flask_jsonpify import jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)
input_tensor_name = ''
output_tensor_name = ''
tf_session = None
model_dir = "models/use/1"

# Read configuration.
# See https://flask.palletsprojects.com/en/1.1.x/config/#configuring-from-files
if os.environ.get('POLISERV_CONFIG') is not None:
    app.config.from_envvar('POLISERV_CONFIG')
    if app.config['MODEL_DIR'] is not None:
        model_dir = app.config['MODEL_DIR']
        print('Using model from ', model_dir)


def extract_tensors(signature_def, graph):
    output = dict()
    for key in signature_def:
        value = signature_def[key]
        if isinstance(value, tf.TensorInfo):
            output[key] = graph.get_tensor_by_name(value.name)
    return output


def extract_tensor_name(signature_def, graph, kind):
    if kind == 'input':
        tensors = extract_tensors(signature_def['serving_default'].inputs, graph)
    elif kind == 'output':
        tensors = extract_tensors(signature_def['serving_default'].outputs, graph)
    else:
        raise ValueError("invalid 'kind' argument")
    # Assuming one tensor on input and output.
    key = list(tensors.keys())[0]
    return tensors.get(key).name


def load_tf_model(model_dir):
    """ Create TF session and load the model.

    :param model_dir: the path to the saved TF model
    """
    global tf_session
    global input_tensor_name, output_tensor_name

    if tf_session is None:
        tf_session = tf.Session(graph=tf.Graph())
        m = tf.saved_model.load(tf_session, tags={'serve'}, export_dir=model_dir)

        input_tensor_name = extract_tensor_name(m.signature_def, tf_session.graph, 'input')
        output_tensor_name = extract_tensor_name(m.signature_def, tf_session.graph, 'output')


def politeness(instances):
    """ Get politeness score for the given text.

    :param instances: the input to the model, there must be at least 2 elements
    :return: a numpy array of politeness scores, i.e. a 2 element vector for each text,
    where the 1st element of the vector is the degree of politeness from 0.0 (impolite)
    to 1.0 (polite).
    """
    assert tf_session is not None
    # There is an issue: a single element array raises the exception:
    # tensorflow.python.framework.errors_impl.InvalidArgumentError: input must be a vector, got shape: []
    # As a workaround please pass at least 2 text instances.
    input_array = np.array(instances, dtype=object)[:, np.newaxis]
    result = tf_session.run([output_tensor_name], feed_dict={input_tensor_name: input_array})
    return result[0]


@app.route('/<version>/models/<name>', methods=['POST'])
def handle_post(version, name):
    data = request.json
    #TODO check version and perform tighter input validation.
    if data is not None \
            and data['instances'] is not None \
            and data['signature_name'] == 'serving_default'\
            and len(data['instances']) > 1:
        result = politeness(data['instances'])
        result_dict = {'predictions': result.tolist()}
        return jsonify(result_dict)
    else:
        return make_response(jsonify({'error': 'Bad request'}), 400)


# Load the model at startup.
load_tf_model(model_dir)

# Note: export FLASK_ENV=development for flask debug mode.
if not app.config['DEBUG'] and __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=8501)
elif __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True, use_reloader=False)

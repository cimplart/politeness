
"""
Train a model based on the Universal Sentence Encoder using the Stanford politeness corpus data.
Save the trained model to HDF5 file.

"""

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import tensorflow as tf

# %%
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks as CB
from sklearn.metrics import confusion_matrix
import logging

# Suppress TF warnings.
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# %%
path = "~/projects/Stanford_politeness_corpus"
data = pd.read_csv(path + "/" + "wikipedia.annotated.csv")

# %%
data = data[["Request", "Normalized Score"]]
print("data length:", len(data))

# %%
# Correct data based on some linguistic features.
data.loc[data['Request'].str.contains("Could you please")
         & (data['Normalized Score'] < 1.0), 'Normalized Score'] = 1.0
data.loc[data['Request'].str.contains("Can you please")
         & (data['Normalized Score'] < 1.0), 'Normalized Score'] = 1.0
data.loc[data['Request'].str.contains("Would you please")
         & (data['Normalized Score'] < 1.0), 'Normalized Score'] = 1.0
# ...

# %%
# Rename column
data = data.rename(columns={"Request": "Request", "Normalized Score": "Score"})
print(data.head())

# %%
q3c = data["Score"].quantile(q=[0.3, 0.7])
print(q3c.values)

# %%
data = data[(data["Score"] < q3c.values[0]) | (data["Score"] > q3c.values[1])]

# %%
text = data["Request"].tolist()
text = np.array(text, dtype=object)[:, np.newaxis]

# %%
def score_to_label(score):
    if (score < 0):
        return(True)   # Impolite
    else:
        return(False)


data_labeled = data["Score"].apply(score_to_label)
label = np.asarray(pd.get_dummies(data_labeled.values), dtype=np.int8)
print(label)

# %%
msk = np.random.rand(len(text)) < 0.95
train_text = text[msk]
test_text = text[~msk]

# %%
train_label = label[msk]
test_label = label[~msk]

# %%
assert len(train_text) == len(train_label)
assert len(test_text) == len(test_label)

# %%
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)


# %%
def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)),
                 signature="default", as_dict=True)["default"]


# %%
embed_size = 512
category_counts = 2
input_text = layers.Input(shape=(1,), dtype=tf.string)
embedding = layers.Lambda(UniversalEmbedding,
                          output_shape=(embed_size,))(input_text)
dense1 = layers.Dense(512, activation='relu')(embedding)
reg1 = layers.Dropout(0.25)(dense1)
dense2 = layers.Dense(256, activation='sigmoid')(reg1)
reg2 = layers.Dropout(0.25)(dense2)
pred = layers.Dense(category_counts, activation='softmax')(reg2)
model = Model(inputs=[input_text], outputs=pred)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()

# %%
mc = tf.keras.callbacks.ModelCheckpoint('./model.h5', verbose=1, save_best_only=True,
                                        monitor='val_acc')

# %%
es = CB.EarlyStopping(monitor='val_acc', mode='auto', verbose=1, patience=4,
                      restore_best_weights=True)

# %%
with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    history = model.fit(train_text, train_label, validation_split=0.25,
                        epochs=5,
                        batch_size=32,
                        verbose=1,
                        callbacks=[es, mc])

# %%
# Check the performance on the test set.
with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    model = tf.keras.models.load_model('./model.h5')
    predicts = model.predict(test_text, batch_size=32)

predicted = predicts.argmax(axis=1)
print(predicted)


# %%
truth = test_label[:, 1]
truth

# %%
m = confusion_matrix(truth, predicted)
print(m)

# %%
# Get the accuracy for the test set.
accuracy = (truth == predicted).sum() / float(len(truth))
print(accuracy)

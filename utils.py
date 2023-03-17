import time
from tensorflow import keras

class time_callback(keras.callbacks.Callback):
    def __init__(self):
        self.start_time = 0
        self.batch_time = []

    def on_train_batch_begin(self, batch, logs=None):
        self.start_time = time.time()
    
    def on_train_batch_end(self, batch, logs=None):
        self.batch_time.append(time.time() - self.start_time)

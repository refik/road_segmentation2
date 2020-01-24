import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score


class GlobalF1(Callback):
    def __init__(self):
        super().__init__()
        self.f1s = []

    def on_epoch_end(self, batch, logs={}):
        predict = np.asarray(self.model.predict(self.validation_data[0]))
        predict = np.argmax(predict, axis=1)
        targ = self.validation_data[1]
        targ = np.argmax(targ, axis=1)
        # self.f1s=f1(targ, predict)
        self.f1s.append(f1_score(targ, predict))
        print(f"F1 score on validation set {self.f1s[-1]}")
        return


class GlobalF1OneToOne(Callback):
    def __init__(self, road_threshold=0.5):
        super().__init__()
        self.f1s = []
        self.road_threshold = road_threshold

    def on_epoch_end(self, batch, logs={}):
        imgs = self.validation_data[0]
        gts = self.validation_data[1]
        predict = np.asarray(self.model.predict(imgs))
        predict = (predict.reshape(-1) > self.road_threshold) * 1
        targ = gts
        targ = (targ.reshape(-1) > self.road_threshold) * 1
        # self.f1s=f1(targ, predict)
        self.f1s.append(f1_score(targ, predict))
        print(f"F1 score on validation set {self.f1s[-1]}")
        return

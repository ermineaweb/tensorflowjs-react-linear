import * as tf from "@tensorflow/tfjs";

export const OPTIMIZERS = {
  sgd: { libelle: "sgd", fn: (lr) => tf.train.sgd(lr) },
  adam: { libelle: "adam", fn: (lr) => tf.train.adam(lr) },
  adagrad: { libelle: "adagrad", fn: (lr) => tf.train.adagrad(lr) },
  adadelta: { libelle: "adadelta", fn: (lr) => tf.train.adadelta(lr) },
  momentum: { libelle: "momentum", fn: (lr) => tf.train.momentum(lr, 1) },
  rmsprop: { libelle: "rmsprop", fn: (lr) => tf.train.rmsprop(lr) },
};

import * as tf from "@tensorflow/tfjs";
import { OPTIMIZERS } from "./optimizers";

export function generateData() {
  const input = tf.tensor([0, 2, 4, 7, 10, 20, 50], [7, 1]);
  const label = tf.tensor([5, 9, 13, 19, 25, 45, 105], [7, 1]);
  return [input, label];
}

export function createModel({
  units = 1,
  learningRate = 0.01,
  optimizer = "adam",
}) {
  const selectOptimizer = (optimizer) => {
    return OPTIMIZERS[optimizer].fn(learningRate);
  };

  const model = tf.sequential();
  model.add(tf.layers.dense({ units, inputShape: [1] }));
  model.compile({
    optimizer: selectOptimizer(optimizer),
    loss: "meanSquaredError",
  });
  return model;
}

export async function trainModel(model, input, label, epochs = 150) {
  await model.fit(input, label, { epochs });
}

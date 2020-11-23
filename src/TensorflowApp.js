import React, { useEffect, useReducer, useState } from "react";
import { createModel, generateData, trainModel } from "./tensor";
import { OPTIMIZERS } from "./optimizers";
import * as tf from "@tensorflow/tfjs-core";

const initialModelOptions = {
  optimizer: "adam",
  learningRate: 0.01,
  epochs: 50,
};

function modelReducer(state, action) {
  switch (action.type) {
    case "SET_OPTIMIZER":
      return { ...state, optimizer: action.value };
    case "SET_LEARNINGRATE":
      return { ...state, learningRate: parseFloat(action.value) };
    case "SET_EPOCHS":
      return { ...state, epochs: parseInt(action.value) };
    default:
      return state;
  }
}

function TensorflowApp() {
  const [input, label] = generateData();
  const [modelOptions, dispatch] = useReducer(
    modelReducer,
    initialModelOptions
  );
  const [modelReady, setModelReady] = useState(true);
  const [target, setTarget] = useState(0);
  const [prediction, setPrediction] = useState("");
  const [targetPredict, setTargetPredict] = useState(target * 2 + 5);
  const [model, setModel] = useState(createModel(modelOptions));

  const handleChangeTarget = (e) => {
    setTarget(Number(e.target.value));
  };

  const handleChangeModelOptions = (action) => (e) => {
    dispatch({ type: action, value: e.target.value });
  };

  const handleTrain = () => {
    setModelReady(false);
    trainModel(model, input, label, modelOptions.epochs).then(() =>
      setModelReady(true)
    );
  };

  useEffect(() => {
    setTarget(0);
    setPrediction("0");
    setModel(createModel(modelOptions));
  }, [modelOptions]);

  useEffect(() => {
    setTargetPredict(target * 2 + 5);
    model
      .predict(tf.tensor([target]))
      .data()
      .then((data) => setPrediction(parseFloat(data).toFixed(2).toString()));
  }, [target]);

  return (
    <div>
      <h1>Tensorflowjs in React</h1>

      <section>
        <h2>1. Data</h2>
        <table>
          <tbody>
            <tr>
              <td>input</td>
              {input.arraySync().map((x) => (
                <td key={x}>{x}</td>
              ))}
            </tr>
            <tr>
              <td>label</td>
              {label.arraySync().map((y) => (
                <td key={y}>{y}</td>
              ))}
            </tr>
          </tbody>
        </table>
      </section>

      <section>
        <h2>2. Model</h2>
        <label htmlFor="optiFn">Optimizer function</label>
        <select
          id="optiFn"
          value={modelOptions.optimizer}
          onChange={handleChangeModelOptions("SET_OPTIMIZER")}
        >
          {Object.keys(OPTIMIZERS).map((key) => (
            <option key={key} value={OPTIMIZERS[key].libelle}>
              {OPTIMIZERS[key].libelle}
            </option>
          ))}
        </select>

        <label htmlFor="learningRate">Learning rate</label>
        <input
          id="learningRate"
          value={modelOptions.learningRate}
          onChange={handleChangeModelOptions("SET_LEARNINGRATE")}
          disabled={!modelReady}
          type="number"
        />
      </section>

      <section>
        <h2>3. Training</h2>
        <label htmlFor="epochs">Epochs</label>
        <input
          id="epochs"
          value={modelOptions.epochs}
          onChange={handleChangeModelOptions("SET_EPOCHS")}
          disabled={!modelReady}
          type="number"
        />
        <div>{`${modelReady ? "Model is ready" : "Model is training..."}`}</div>
        <button onClick={handleTrain} disabled={!modelReady}>
          Train model
        </button>
      </section>

      <section>
        <h2>4. Predict</h2>
        <label htmlFor="target">Input</label>
        <input
          id="target"
          value={target}
          onChange={handleChangeTarget}
          disabled={!modelReady}
          type="number"
        />
      </section>

      <section>
        <h2>5. Results</h2>
        <p>
          Expected result : {`${target} * 2 + 5 = `}
          <strong>{targetPredict}</strong>
        </p>
        <p>
          Predicted result :<strong> {prediction} </strong>
        </p>
      </section>

      <section>
        <a href={"https://github.com/ermineaweb/tensorflowjs-react-linear"}>
          Sources
        </a>
        <a href={"https://romainamichaud.com"}>My website</a>
      </section>
    </div>
  );
}

export default TensorflowApp;

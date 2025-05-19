[Recorded by: 2025-05-19]

When model overfitting, I need some tools to check the model performance and need some techniques to prevent overfitting.

### TODO List

- [ ] Implement the `Dropout` layer
- [ ] Implement the `BatchNormalization` layer
- [ ] Enhance the `EarlyStopping` callback
    - The `EarlyStopping` callback should monitor the validation loss and stop the training if the validation loss does not improve for a certain number of epochs.
    - The `EarlyStopping` callback should also save the best model weights.
- Implement others performance metrics
    - [x] `accuracy`
    - [x] `confusion matrix`
    - [ ] `precision`
    - [ ] `recall`
    - [ ] `f1-score`
- [ ] Implement the `Tuner` class
    - The `Tuner` class should be used to tune the hyperparameters of the model.
    - The `Tuner` class should support the following hyperparameters:
        - learning rate
        - batch size
        - number of epochs
        - optimizer
        - loss function
        - model architecture
    - The `Tuner` class should support the following tuning methods:
        - grid search
        - random search
        - Bayesian optimization
- [ ] Support callback function for MLflow api

[Recorded by: 2025-05-15]

### TODO List

- [ ] Move the `split` functions to the another repository
    - The `split` functions are used to split the dataset into training and validation sets.
    - It is not a part of the neural network framework, so it should be moved to another repository.
- [ ] Support `model.train()` and `model.eval()` mode as the PyTorch
    - The `train` mode should enable the dropout and batch normalization layers.
    - The `eval` mode should disable the dropout and batch normalization layers.
    - For the advanced nn models, the `train` and `eval` modes is nessary.
        - RNN, LSTM, GRU, Transformer, etc.

[Recorded by: 2025-05-12]

### TODO List

- [ ] Refactor the `core` folder structure to separate by category
    - `core` folder should be separated into `optimizers`, `layers`, `losses`, `utils`, and `callbacks`

[Recorded by: 2025-05-08]

### TODO List
- [ ] Implement the `Adam` optimizer
- [ ] ~~Implement the `LabelEncoder` and `OneHotEncoder` for the data loader~~
    - by the desgin mindset, the nn framework should not be responsible for the data preprocessing

[Recorded by: 2025-05-04]

### TODO List
- [ ] Add test cases for the data loader
- [ ] Add test cases for the trainer
    - consider to embed the trainer into the model as `model.fit()`
- [x] Implement the `logger` and support output format CSV, JSON, and tensorboard
- [x] Implement the `EarlyStopping` callback
- [ ] Implement the `ModelCheckpoint` callback
- [ ] Implement the `gradient clipping` in the trainer
- [x] Implement the utility functions to plot the training and validation loss
- [ ] ~~Implement the `data augmentation` for the data loader~~
    - by the desgin mindset, the nn framework should not be responsible for the data preprocessing
- [ ] ~~Implement the `data normalization` for the data loader~~
    - by the desgin mindset, the nn framework should not be responsible for the data preprocessing
- [ ] Add notebook to explain how backpropagation, auto grad and the computation graph works
- [ ] Add note to explain how importain the `command pattern` and `responsibility chain pattern` are in the framework design

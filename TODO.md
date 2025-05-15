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
- [ ] Implement the `logger` and support output format CSV, JSON, and tensorboard
- [ ] Implement the `EarlyStopping` callback
- [ ] Implement the `ModelCheckpoint` callback
- [ ] Implement the `gradient clipping` in the trainer
- [ ] Implement the utility functions to plot the training and validation loss
- [ ] ~~Implement the `data augmentation` for the data loader~~
    - by the desgin mindset, the nn framework should not be responsible for the data preprocessing
- [ ] ~~Implement the `data normalization` for the data loader~~
    - by the desgin mindset, the nn framework should not be responsible for the data preprocessing
- [ ] Add notebook to explain how backpropagation, auto grad and the computation graph works
- [ ] Add note to explain how importain the `command pattern` and `responsibility chain pattern` are in the framework design

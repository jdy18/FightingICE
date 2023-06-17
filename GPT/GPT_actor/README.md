# README

- Install hugging face before running.

  ```
  pip install transformers
  ```

- GPT model is an autoregressive model, based on transformer decoder. That means the model only attends to the states before current state.

- There four hyperparameters:

  - input_size: After encoded by Mel-encoder, the dimension is 2560. If you want to concatenate state and action, please replace it with 2560+40=2600.

  - n_embd: embedding size.

  - n_layer: the number of transformer layer.

  - n_head: the number of self attention heads.

    change n_embd, n_layer and n_head to improve computational efficiency.
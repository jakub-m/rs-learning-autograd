# Automatic gradient in Rust

My learning exercise to understand automatic gradient, and to exercise Rust.

## Learnings, observations

- Computation graph can be agnostic to the underlying representation of the value (if e.g. it is f32 or int).
- `From` tuple to Composite simplifies syntax
- Rust forces you to separate the concepts, like expression builder from computation, otherwise the problem
  becomes to entangled. When separated, the problems can be nicely implemented.

## Materials

- [Hugging Face: What's Automatic Differentiation?](https://huggingface.co/blog/andmholm/what-is-automatic-differentiation)
- [YT: What is Automatic Differentiation?](https://www.youtube.com/watch?v=wG_nF1awSSY)
- [Demystifying AutoGrad in Machine Learning](https://medium.com/@weidagang/demystifying-autograd-in-machine-learning-eb7d5c875ff2)

## Possible TODOs

- Compute partial derivates and outputs for all inputs at once.
- Implement reverse mode.
- Implement gradient descent with this autograd
- Implement larger test cases (examples from YT and HF)
- Train something

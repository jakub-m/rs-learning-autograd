# Automatic gradient in Rust

A learning exercise to understand automatic gradient, and to exercise Rust. The intention is to have a fully working minimal autograd library that will be able to do automatic differentiation and training.

## Learnings, observations

- Computation graph can be agnostic to the underlying representation of the value (if e.g. it is f32 or int).
- `From` tuple to Composite simplifies syntax
- Rust forces you to separate the concepts, like expression builder from computation, otherwise the problem becomes to
  entangled. When separated, the problems can be nicely implemented.
- You can have optional arguments in a function with `&[Opts]`, where `Opts` is an enum.
- Fitting parameters of a polynomial with a gradient descent is hard, possibly (my naive interpretation) because small
  changes in the parameters can cause a large change at the output.

## Materials

### Automatic gradient

- [Hugging Face: What's Automatic Differentiation?](https://huggingface.co/blog/andmholm/what-is-automatic-differentiation)
- [YT: What is Automatic Differentiation?](https://www.youtube.com/watch?v=wG_nF1awSSY)
- [Demystifying AutoGrad in Machine Learning](https://medium.com/@weidagang/demystifying-autograd-in-machine-learning-eb7d5c875ff2)

### Derivatives

- [Max pooling and backprop](https://datascience.stackexchange.com/a/11703)

## Using [marimo][ref_marimo] to view test outputs

There are integration tests that compare a function and that function re-created using derivatives. If the test fails,
then a CSV file is created showing the difference. You can view the files with [marimo][ref_marimo] notebook:

```bash
make marimo
```

[ref_marimo]: https://marimo.io/

## Possible TODOs

- Add f32 to matrix type
- Implement larger test cases (examples from YT and HF)
- Train something
- Figure that `(a-b)*(a-b)` are the same nodes, not two different ones. To do that, use hash of node, instead of incremental id.

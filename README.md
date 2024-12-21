# Automatic gradient in Rust

A learning exercise to understand automatic gradient, and to exercise Rust.

## Learnings, observations

- Computation graph can be agnostic to the underlying representation of the value (if e.g. it is f32 or int).
- `From` tuple to Composite simplifies syntax
- Rust forces you to separate the concepts, like expression builder from computation, otherwise the problem becomes to
  entangled. When separated, the problems can be nicely implemented.
- You can have optional arguments in a function with `&[Opts]`, where `Opts` is an enum.

## Materials

### Automatic gradient

- [Hugging Face: What's Automatic Differentiation?](https://huggingface.co/blog/andmholm/what-is-automatic-differentiation)
- [YT: What is Automatic Differentiation?](https://www.youtube.com/watch?v=wG_nF1awSSY)
- [Demystifying AutoGrad in Machine Learning](https://medium.com/@weidagang/demystifying-autograd-in-machine-learning-eb7d5c875ff2)

## Using [marimo][ref_marimo] to view test outputs

There are integration tests that compare a function and that function re-created using derivatives. If the test fails,
then a CSV file is created showing the difference. You can view the files with [marimo][ref_marimo] notebook:

```bash
make marimo
```

[ref_marimo]: https://marimo.io/

## Possible TODOs

- Implement gradient descent with this autograd
- Implement larger test cases (examples from YT and HF)
- Train something

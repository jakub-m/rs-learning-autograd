# Automatic gradient in Rust

My learning exercise to understand automatic gradient, and to exercise Rust.

## Learnings, observations

- Computation graph can be agnostic to the underlying representation of the value (if e.g. it is f32 or int).

## Materials

- [What is Automatic Differentiation?](https://www.youtube.com/watch?v=wG_nF1awSSY)
- [Demystifying AutoGrad in Machine Learning](https://medium.com/@weidagang/demystifying-autograd-in-machine-learning-eb7d5c875ff2)

## Possible TODOs

- Compute partial derivates and outputs for all inputs at once.
- Implement reverse mode.
- Do not use enum but a reference to trait.
- Implement gradient descent with this autograd
- Train something

use std::hash::Hash;
use std::{collections::HashMap, ops};

type F = f32;

/// Expr is an enum so we can use it in HashMap in the State. The big drawback is that the enum is "closed"
/// in a sense that it needs to know all the allowed expressions upfront.
#[derive(PartialEq, Eq, Hash)]
pub enum Expr<'a> {
    /// A named variable.
    Variable(String),
    Add(&'a Expr<'a>, &'a Expr<'a>),
    Sub(&'a Expr<'a>, &'a Expr<'a>),
    Div(&'a Expr<'a>, &'a Expr<'a>),
    Mul(&'a Expr<'a>, &'a Expr<'a>),
    Sin(&'a Expr<'a>),
    Exp(&'a Expr<'a>),
}

impl<'a, 'b> Expr<'a>
where
    'a: 'b, // The Expr ('a) needs to leave at least as long as the State ('b) that references that expression.
{
    /// Return value of the expression, and cache the value in the State. This way the same
    /// expression is not computed twice in the computing graph.
    pub fn forward(&'a self, state: &'b mut State<'a>) -> Result<F, String> {
        if let Some(value) = state.forward_values.get(self) {
            Ok(*value)
        } else {
            let (result, dot) = match self {
                Expr::Variable(name) => Err(format!("Variable {} is not set", name)),
                Expr::Add(lhs, rhs) => {
                    let ((a, a_dot), (b, b_dot)) = get_fd2(state, lhs, rhs)?;
                    Ok(((a + b), (a_dot + b_dot)))
                }
                Expr::Sub(lhs, rhs) => {
                    let ((a, a_dot), (b, b_dot)) = get_fd2(state, lhs, rhs)?;
                    Ok(((a - b), (a_dot - b_dot)))
                }
                Expr::Div(lhs, rhs) => {
                    let ((a, a_dot), (b, b_dot)) = get_fd2(state, lhs, rhs)?;
                    let d = (a_dot * b - a * b_dot) / (b * b);
                    Ok(((a / b), d))
                }
                Expr::Mul(lhs, rhs) => {
                    let ((a, a_dot), (b, b_dot)) = get_fd2(state, lhs, rhs)?;
                    let d = a * b_dot + a_dot * b;
                    Ok(((a * b), d))
                }
                Expr::Sin(arg) => {
                    let arg = arg.forward(state)?;
                    Ok(((arg.sin()), 0.0)) // todo
                }
                Expr::Exp(arg) => {
                    let arg = arg.forward(state)?;
                    Ok(((arg.exp()), 0.0)) // todo
                }
            }?;
            state.set_expr_value(self, result);
            Ok(result)
        }
    }

    /// Return cached derivate. One need to first run .forward to calculate the proper state.
    pub fn dot(&'a self, state: &'b mut State<'a>) -> Result<F, String> {
        //
        //todo!()
        Ok(0.0)
    }
}

/// A helper function to get forward and dot values for an expression.
fn get_fd1<'a, 'b>(state: &'a mut State<'b>, v1: &'b Expr) -> Result<(F, F), String> {
    let f1 = v1.forward(state)?;
    let d1 = v1.dot(state)?;
    Ok((f1, d1))
}
fn get_fd2<'a, 'b>(
    state: &'a mut State<'b>,
    v1: &'b Expr,
    v2: &'b Expr,
) -> Result<((F, F), (F, F)), String> {
    let f1 = v1.forward(state)?;
    let d1 = v1.dot(state)?;
    let f2 = v2.forward(state)?;
    let d2 = v2.dot(state)?;
    Ok(((f1, d1), (f2, d2)))
}

struct State<'a> {
    forward_values: HashMap<&'a Expr<'a>, F>,
    //dot_values: HashMap<&'a Expr<'a>, F>,
}

impl<'a> State<'a> {
    pub fn new() -> State<'a> {
        State {
            forward_values: HashMap::new(),
        }
    }
    pub fn set_expr_value(&mut self, expr: &'a Expr, value: F) {
        self.forward_values.insert(expr, value);
    }
}

/// An input variable, with its name. Two variables with the same name are the same variable. An utility shim over Expr
/// So the user does not need to operate on Expr.
struct Variable(String);

impl Variable {
    fn new(name: &str) -> Variable {
        Variable(name.to_owned())
    }

    fn name(&self) -> &str {
        return self.0.as_str();
    }
}

impl<'a> From<&'a Variable> for Expr<'a> {
    fn from(variable: &'a Variable) -> Self {
        Expr::Variable(variable.name().to_owned())
    }
}

impl<'a> ops::Add<&'a Expr<'a>> for &'a Expr<'a> {
    type Output = Expr<'a>;

    fn add(self, rhs: &'a Expr) -> Self::Output {
        Expr::Add(self, rhs)
    }
}

impl<'a> ops::Sub<&'a Expr<'a>> for &'a Expr<'a> {
    type Output = Expr<'a>;

    fn sub(self, rhs: &'a Expr) -> Self::Output {
        Expr::Sub(self, rhs)
    }
}

impl<'a> ops::Div<&'a Expr<'a>> for &'a Expr<'a> {
    type Output = Expr<'a>;

    fn div(self, rhs: &'a Expr) -> Self::Output {
        Expr::Div(self, rhs)
    }
}

impl<'a> ops::Mul<&'a Expr<'a>> for &'a Expr<'a> {
    type Output = Expr<'a>;

    fn mul(self, rhs: &'a Expr) -> Self::Output {
        Expr::Mul(self, rhs)
    }
}

pub fn sin<'a>(term: &'a Expr<'a>) -> Expr<'a> {
    Expr::Sin(term)
}

pub fn exp<'a>(term: &'a Expr<'a>) -> Expr<'a> {
    Expr::Exp(term)
}

//pub struct ExpTerm<'a>(&'a dyn Expr);
//
//impl<'a> Expr for ExpTerm<'a> {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test from
    /// https://youtu.be/wG_nF1awSSY?si=kd5Ny4055k3mXc8r&t=366
    ///```
    ///f(x1, x2) = [sin(x1/x2) + x1/x2 - exp(x2)] x [x1/x2-exp(x2)]
    ///  vm1 v0         --v1-    -v1-                --v1-
    ///            ----v2----           --v3----           --v3---
    ///                          -------v4------     -----v4------
    ///              --------------v5----------
    ///            --------------------------------v6--------------
    ///```
    #[test]
    fn yt1() {
        let x1 = Variable::new("x1");
        let vm1: Expr = (&x1).into();
        let x2 = Variable::new("x2");
        let v0: Expr = (&x2).into();
        let v1 = &vm1 / &v0;
        let v2 = sin(&v1);
        let v3 = exp(&v0);
        let v4 = &v1 - &v3;
        let v5 = &v2 + &v4;
        let v6 = &v5 * &v4;

        let mut state = State::new();
        state.set_expr_value(&vm1, 1.5); // TODO how to use x1?
        state.set_expr_value(&v0, 0.5);
        assert_eq!(v1.forward(&mut state), Ok(3.0));
        assert_almost_eq(v2.forward(&mut state).unwrap(), 0.141);
        assert_almost_eq(v3.forward(&mut state).unwrap(), 1.649);
        assert_almost_eq(v4.forward(&mut state).unwrap(), 1.351);
        assert_almost_eq(v5.forward(&mut state).unwrap(), 1.492);
        assert_almost_eq(v6.forward(&mut state).unwrap(), 2.017);

        // todo below
        // todo other tests
        //forward.value();
        //forward.dot();
    }

    fn assert_almost_eq(f1: f32, f2: f32) {
        let eps = 0.001;
        let d = (f1 - f2).abs();
        assert!(d < eps, "{f1} != {f2} (eps={eps})");
    }
}

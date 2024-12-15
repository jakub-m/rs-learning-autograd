use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet};
use std::env::var;
use std::fmt;
use std::ops;

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Hash)]
pub struct Ident(usize);

/// A generic expressionError
//#[derive(Clone, Debug)]
//pub struct ExprError(String);
//
//pub type ExprResult = Result<(), ExprError>;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Node {
    Variable(Ident),
    Add(Ident, Ident),
    Mul(Ident, Ident),
}

/// An identifier coupled with a reference to ExprBuilder, so it can be later used in further arithmetic operations.
#[derive(Clone, Copy, Debug)]
pub struct Expr<'a> {
    // eb cannot be mut because mut is not Clone, therefore is not Copy, and we want Copy.
    eb: &'a ExprBuilder,
    ident: Ident,
}

impl<'a> Expr<'a> {
    fn fmt_node(&self, f: &mut fmt::Formatter<'_>, node: &Node) -> fmt::Result {
        let id_to_node = self.eb.id_to_node.borrow();
        match node {
            Node::Variable(ident) => {
                let name = self
                    .eb
                    .id_to_name
                    .borrow()
                    .get(ident)
                    .expect(format!("Variable with {:?} does not exist", ident).as_str())
                    .to_owned();
                write!(f, "{}", name)?;
            }
            Node::Add(ident1, ident2) => {
                let node1 = id_to_node.get(ident1).ok_or(fmt::Error)?;
                let node2 = id_to_node.get(ident2).ok_or(fmt::Error)?;
                write!(f, "(+ ")?;
                self.fmt_node(f, &node1)?;
                write!(f, " ")?;
                self.fmt_node(f, &node2)?;
                write!(f, ")")?;
            }
            Node::Mul(ident1, ident2) => {
                let node1 = id_to_node.get(ident1).ok_or(fmt::Error)?;
                let node2 = id_to_node.get(ident2).ok_or(fmt::Error)?;
                write!(f, "(* ")?;
                self.fmt_node(f, &node1)?;
                write!(f, " ")?;
                self.fmt_node(f, &node2)?;
                write!(f, ")")?;
            }
        };
        Ok(())
    }

    pub fn ident(&self) -> Ident {
        self.ident
    }
}

impl<'a> fmt::Display for Expr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let map = self.eb.id_to_node.borrow();
        let node = map.get(&self.ident).ok_or(fmt::Error)?;
        self.fmt_node(f, &node)
    }
}

#[derive(Debug)]
struct ExprBuilder {
    /// The map contains expression trees with references.
    id_to_node: RefCell<BTreeMap<Ident, Node>>,
    id_to_name: RefCell<BTreeMap<Ident, String>>,
    name_set: RefCell<HashSet<String>>,
}

impl<'a> ExprBuilder {
    pub fn new() -> ExprBuilder {
        ExprBuilder {
            id_to_node: RefCell::new(BTreeMap::new()),
            id_to_name: RefCell::new(BTreeMap::new()),
            name_set: RefCell::new(HashSet::new()),
        }
    }

    pub fn new_variable(&'a self, name: &str) -> Expr<'a> {
        let ident = self.new_ident();

        let mut id_to_name = self.id_to_name.borrow_mut();
        if let Some(old_name) = id_to_name.insert(ident, name.to_owned()) {
            panic!(
                "Variable for  {:?} already exists with name {}",
                ident, old_name
            );
        }

        let mut name_set = self.name_set.borrow_mut();
        if !name_set.insert(name.to_owned()) {
            panic!("Variable with name {} already exists", name)
        }

        let node = Node::Variable(ident);
        let mut id_to_node = self.id_to_node.borrow_mut();
        id_to_node.insert(ident, node);
        Expr { eb: &self, ident }
    }

    /// register is a mutable operation on self.map. `register` is not explicitly mut, to allow Copy and
    /// ergonomic arithmetic syntax.
    fn register(&self, node: Node) -> Ident {
        let ident = self.new_ident();
        let mut map = self.id_to_node.borrow_mut();
        map.insert(ident, node);
        ident
    }

    fn new_ident(&self) -> Ident {
        Ident(self.id_to_node.borrow().len())
    }
}

pub struct ComputeGraph {
    id_to_node: RefCell<BTreeMap<Ident, Node>>,
    id_to_name: RefCell<BTreeMap<Ident, String>>,
}

impl ComputeGraph {
    pub fn set_variable<F>(&self, state: &mut State<F>, ident: &Ident, value: F) {
        let id_to_node = self.id_to_node.borrow();
        let node = id_to_node
            .get(ident)
            .expect(format!("No such ident {:?}", ident).as_ref());
        if let Node::Variable(got_ident) = node {
            if ident != got_ident {
                panic!("Idents not equal, a bug?")
            }
        } else {
            panic!("Ident is not a variable");
        }
        if let Some(_) = state.primals.insert(*ident, value) {
            panic!("Ident already set")
        }
    }

    pub fn compute<'a, F>(&self, ident: &Ident, state: &'a mut State<F>) -> &'a F
    where
        F: ops::Add<Output = F> + Copy,
    {
        if let Some(value) = state.primals.get(ident) {
            return value;
        };
        //let id_to_node = self.id_to_node.borrow();
        //let node = id_to_node.get(ident).expect("No such ident");
        //// TODO change Add, Mul, etc to Ary0, Ary1 and Ary2 nodes.
        //let value = match node {
        //    Node::Variable(variable_ident) => {
        //        assert_eq!(ident, variable_ident);
        //        *state
        //            .primals
        //            .get(ident)
        //            .expect("Variable ident not in primals")
        //    }
        //    Node::Add(ident1, ident2) => {
        //        let value1 = self.compute(ident1, state);
        //        let value2 = self.compute(ident2, state);
        //        (*value1 + *value2)
        //    }
        //    Node::Mul(ident1, ident2) => todo!(),
        //};
        todo!()
        //
    }
}

impl ExprBuilder {
    pub fn freeze(self) -> ComputeGraph {
        ComputeGraph {
            id_to_node: self.id_to_node,
            id_to_name: self.id_to_name,
        }
    }
}

impl<'a> ops::Add for Expr<'a> {
    type Output = Expr<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        let node = Node::Add(self.ident, rhs.ident);
        let ident = self.eb.register(node);
        Expr { ident, eb: self.eb }
    }
}

impl<'a> ops::Mul for Expr<'a> {
    type Output = Expr<'a>;

    fn mul(self, rhs: Self) -> Self::Output {
        let node = Node::Mul(self.ident, rhs.ident);
        let ident = self.eb.register(node);
        Expr { ident, eb: self.eb }
    }
}

pub struct State<F> {
    pub(crate) primals: BTreeMap<Ident, F>,
}

impl<F> State<F> {
    pub fn new() -> State<F> {
        State {
            primals: BTreeMap::new(),
        }
    }

    pub fn get(&self, ident: &Ident) -> Option<&F> {
        self.primals.get(ident)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syntax() {
        let eb = ExprBuilder::new();
        let x1 = eb.new_variable("x1");
        let x2 = eb.new_variable("x2");
        let x3 = x1 + x2;
        let x4 = x1 + x2;
        let z = x1 + x2 * x3 + x4;
        assert_eq!("(+ (+ x1 (* x2 (+ x1 x2))) (+ x1 x2))", format!("{}", z));
    }

    #[test]
    fn forward_add_mul() {
        let eb = ExprBuilder::new();
        let x1 = eb.new_variable("x1");
        let x2 = eb.new_variable("x2");
        let y = x1 + x2 * x1 + x2;

        // Operate on frozen graph.
        let x1 = &x1.ident();
        let x2 = &x2.ident();
        let y = &y.ident();
        let graph = eb.freeze();
        let mut state = State::<f64>::new();
        graph.set_variable(&mut state, &x1, 2.0);
        graph.set_variable(&mut state, &x2, 3.0);
        graph.compute(y, s);
        //assert_eq!(graph.get_forward(y, s).to_owned(), 11.0);
    }
}

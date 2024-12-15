use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet};
use std::fmt::{self, Display};
use std::ops;

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Hash)]
pub struct Ident(usize);

impl Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Node {
    Const(Ident),
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
            Node::Const(ident) => {
                let name = self
                    .eb
                    .id_to_name
                    .borrow()
                    .get(ident)
                    .expect(format!("Variable with {} does not exist", ident).as_str())
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

        let node = Node::Const(ident);
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
}

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::ops::Add;

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
pub struct Ident(usize);

#[derive(Debug)]
pub enum Node {
    Variable(Ident),
    Add(Ident, Ident),
}

/// An identifier coupled with a reference to ExprBuilder, so it can be later used in further arithmetic operations.
#[derive(Clone, Copy, Debug)]
pub struct Expr<'a> {
    // eb cannot be mut because mut is not Clone, therefore is not Copy, and we want Copy.
    eb: &'a ExprBuilder,
    ident: Ident,
}

impl<'a> Expr<'a> {
    fn fmt_node_indent(
        &self,
        f: &mut fmt::Formatter<'_>,
        node: &Node,
        indent: usize,
    ) -> fmt::Result {
        //for _ in 0..indent {
        //    write!(f, " ")?
        //}
        let map = self.eb.map.borrow();
        match node {
            Node::Variable(ident) => write!(f, "{}", ident.0)?,
            Node::Add(ident1, ident2) => {
                let node1 = map.get(ident1).ok_or(fmt::Error)?;
                let node2 = map.get(ident2).ok_or(fmt::Error)?;
                write!(f, "(+ ")?;
                self.fmt_node_indent(f, &node1, indent + 1)?;
                write!(f, " ")?;
                self.fmt_node_indent(f, &node2, indent + 1)?;
                write!(f, ")")?;
            }
        };
        Ok(())
    }
}

impl<'a> fmt::Display for Expr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        //self.fmt_indent(f, 0)
        let map = self.eb.map.borrow();
        let node = map.get(&self.ident).ok_or(fmt::Error)?;
        self.fmt_node_indent(f, &node, 0)
    }
}

#[derive(Debug)]
struct ExprBuilder {
    /// The map contains expression trees with references.
    map: RefCell<BTreeMap<Ident, Node>>,
}

impl<'a> ExprBuilder {
    pub fn new() -> ExprBuilder {
        ExprBuilder {
            map: RefCell::new(BTreeMap::new()),
        }
    }

    pub fn new_variable(&'a self) -> Expr<'a> {
        let ident = self.new_ident();
        let node = Node::Variable(ident);
        let mut map = self.map.borrow_mut();
        map.insert(ident, node);
        Expr { eb: &self, ident }
    }

    /// register is a mutable operation on self.map. `register` is not explicitly mut, to allow Copy and
    /// ergonomic arithmetic syntax.
    fn register(&self, node: Node) -> Ident {
        let ident = self.new_ident();
        let mut map = self.map.borrow_mut();
        map.insert(ident, node);
        ident
    }

    pub fn freeze(&self) -> Node {
        todo!();
    }

    fn new_ident(&self) -> Ident {
        Ident(self.map.borrow().len())
    }
}

impl<'a> Add for Expr<'a> {
    type Output = Expr<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        let node = Node::Add(self.ident, rhs.ident);
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
        let x1 = eb.new_variable();
        let x2 = eb.new_variable();
        let x3 = x1 + x2;
        let x4 = x1 + x2;
        let z = x1 + x2 + x3 + x4;
        //assert!(false, "{}", x3);
        assert!(false, "{}", z);
    }
}

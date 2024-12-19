//! Definition of core syntax. The core syntax allows to use expressions like `x + y * z`,
//! and build a computation graph out of those expressions.
use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet};
use std::fmt::{self, Display};

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Hash)]
pub struct Ident(usize);

impl Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_{}", self.0)
    }
}

/// Some nodes are variables, and those variables have names stored aside. VariableNameId
/// points to that unique name. The type is only to distinguish Ident from variable name.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Hash)]
pub struct VariableNameId(Ident);

impl Display for VariableNameId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "name({})", self.0)
    }
}

impl From<Ident> for VariableNameId {
    fn from(value: Ident) -> Self {
        VariableNameId(value)
    }
}

impl From<VariableNameId> for Ident {
    fn from(value: VariableNameId) -> Self {
        value.0
    }
}

impl<'a> From<&'a VariableNameId> for &'a Ident {
    fn from(value: &'a VariableNameId) -> Self {
        &value.0
    }
}

pub trait Operator: Clone + Copy + fmt::Debug + fmt::Display {}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Node<OP2>
where
    OP2: Operator,
{
    Variable(VariableNameId),
    Ary2(OP2, Ident, Ident),
}

/// An identifier coupled with a reference to ExprBuilder, so it can be later used in further arithmetic operations.
#[derive(Clone, Copy, Debug)]
pub struct Expr<'a, OP2>
where
    OP2: Operator,
{
    pub ident: Ident,

    // eb cannot be mut because mut is not Clone, therefore is not Copy, and we want Copy.
    eb: &'a ExprBuilder<OP2>,
}

impl<'a, OP2> Expr<'a, OP2>
where
    OP2: Operator,
{
    pub fn register_and_continue_expr(&self, node: Node<OP2>) -> Expr<'a, OP2> {
        let ident = self.eb.register(node);
        Expr { ident, eb: self.eb }
    }
}

impl<'a, OP2> Expr<'a, OP2>
where
    OP2: Operator,
{
    fn fmt_node(&self, f: &mut fmt::Formatter<'_>, node: &Node<OP2>) -> fmt::Result {
        let id_to_node = self.eb.id_to_node.borrow();
        match node {
            Node::Variable(name_id) => {
                let name = self
                    .eb
                    .id_to_name
                    .borrow()
                    .get(name_id)
                    .expect(format!("Variable with {} does not exist", name_id).as_str())
                    .to_owned();
                write!(f, "{}", name)?;
            }
            Node::Ary2(op, ident1, ident2) => {
                let node1 = id_to_node.get(ident1).ok_or(fmt::Error)?;
                let node2 = id_to_node.get(ident2).ok_or(fmt::Error)?;
                write!(f, "({} ", op)?;
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

impl<'a, OP2> fmt::Display for Expr<'a, OP2>
where
    OP2: Operator,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let map = self.eb.id_to_node.borrow();
        let node = map.get(&self.ident).ok_or(fmt::Error)?;
        self.fmt_node(f, &node)
    }
}

#[derive(Debug)]
pub struct ExprBuilder<OP2>
where
    OP2: Operator,
{
    /// The map contains expression trees with references.
    pub(super) id_to_node: RefCell<BTreeMap<Ident, Node<OP2>>>,
    pub(super) id_to_name: RefCell<BTreeMap<VariableNameId, String>>,
    pub(super) name_set: RefCell<HashSet<String>>,
}

impl<'a, OP2> ExprBuilder<OP2>
where
    OP2: Operator,
{
    pub fn new() -> ExprBuilder<OP2> {
        ExprBuilder {
            id_to_node: RefCell::new(BTreeMap::new()),
            id_to_name: RefCell::new(BTreeMap::new()),
            name_set: RefCell::new(HashSet::new()),
        }
    }

    pub fn new_variable(&'a self, name: &str) -> Expr<'a, OP2> {
        let ident = self.new_ident();
        let name_id: VariableNameId = ident.into();

        let mut id_to_name = self.id_to_name.borrow_mut();
        if let Some(old_name) = id_to_name.insert(name_id, name.to_owned()) {
            panic!(
                "Variable for  {:?} already exists with name {}",
                ident, old_name
            );
        }

        let mut name_set = self.name_set.borrow_mut();
        if !name_set.insert(name.to_owned()) {
            panic!("Variable with name {} already exists", name)
        }

        let node = Node::Variable(name_id);
        let mut id_to_node = self.id_to_node.borrow_mut();
        id_to_node.insert(ident, node);
        Expr { eb: &self, ident }
    }

    /// register is a mutable operation on self.map. `register` is not explicitly mut, to allow Copy and
    /// ergonomic arithmetic syntax.
    fn register(&self, node: Node<OP2>) -> Ident {
        let ident = self.new_ident();
        let mut map = self.id_to_node.borrow_mut();
        map.insert(ident, node);
        ident
    }

    fn new_ident(&self) -> Ident {
        Ident(self.id_to_node.borrow().len())
    }
}

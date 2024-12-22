//! Definition of core syntax. The core syntax allows to use expressions like `x + y * z`,
//! and build a computation graph out of those expressions. The core syntax is generic and does not impose
//! type of variables underlying computation (like f32 vs f64) or what operations are actually implemented (like addition, or logarithm).
use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet};
use std::fmt::{self, Display};
use std::ops;

/// A type of the computed value (like, f32). [ops::Add] is needed so we can update the adjoins.
pub trait ComputValue:
    Clone + fmt::Display + fmt::Debug + DefaultAdjoin + ops::Add<Output = Self>
{
}

/// Returns an initial adjoin for a type (a "1").
pub trait DefaultAdjoin {
    fn default_adjoin() -> Self;
}

/// Identifier of an [Expr][Expr]. Ident is [Copy] so we can have ergonomic syntax of building
/// the expression tree, like `y = a + b`.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Hash)]
pub struct Ident(usize);

impl Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_{}", self.0)
    }
}

/// Some nodes are variables, and those variables have names stored aside. VariableNameId
/// points to that unique name. The type is only to distinguish [Ident] from the variable name.
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

///A node in the expression tree.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Node<F, OP1, OP2>
where
    F: ComputValue,
    OP1: Operator,
    OP2: Operator,
{
    /// A constant value.
    Const(F),
    /// A named variable.
    Variable(VariableNameId),
    /// Arity-1 operation, like ln or sin.
    Ary1(OP1, Ident),
    /// Arity-2 operation, like addition.
    Ary2(OP2, Ident, Ident),
}

/// An identifier coupled with a reference to ExprBuilder, so it can be later used in further arithmetic operations.
/// Expr should be Copy so we can have ergonomic expressions like `y = v1 + v2` without additional `&` or `.clone()`.
#[derive(Clone, Copy, Debug)]
pub struct Expr<'a, F, OP1, OP2>
where
    F: ComputValue,
    OP1: Operator,
    OP2: Operator,
{
    pub ident: Ident, // TODO make ident private? or remove ident() method?

    // eb cannot be mut because mut is not Clone, therefore is not Copy, and we want Copy to be able to do `a + b` on those expressions.
    eb: &'a ExprBuilder<F, OP1, OP2>,
}

impl<'a, F, OP1, OP2> Expr<'a, F, OP1, OP2>
where
    F: ComputValue,
    OP1: Operator,
    OP2: Operator,
{
    pub fn register_and_continue_expr(&self, node: Node<F, OP1, OP2>) -> Expr<'a, F, OP1, OP2> {
        let ident = self.eb.register_node(node);
        Expr { ident, eb: self.eb }
    }
}

impl<'a, F, OP1, OP2> Expr<'a, F, OP1, OP2>
where
    F: ComputValue,
    OP1: Operator,
    OP2: Operator,
{
    fn fmt_node(&self, f: &mut fmt::Formatter<'_>, node: &Node<F, OP1, OP2>) -> fmt::Result {
        let id_to_node = self.eb.id_to_node.borrow();
        match node {
            Node::Const(value) => write!(f, "{}", value)?,
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
            Node::Ary1(op, ident) => {
                let node = id_to_node.get(ident).ok_or(fmt::Error)?;
                write!(f, "{}(", op)?;
                self.fmt_node(f, node)?;
                write!(f, ")",)?;
            }
            Node::Ary2(op, ident1, ident2) => {
                let node1 = id_to_node.get(ident1).ok_or(fmt::Error)?;
                let node2 = id_to_node.get(ident2).ok_or(fmt::Error)?;
                write!(f, "(")?;
                self.fmt_node(f, &node1)?;
                write!(f, "{}", op)?;
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

impl<'a, F, OP1, OP2> fmt::Display for Expr<'a, F, OP1, OP2>
where
    F: ComputValue,
    OP1: Operator,
    OP2: Operator,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let map = self.eb.id_to_node.borrow();
        let node = map.get(&self.ident).ok_or(fmt::Error)?;
        self.fmt_node(f, &node)
    }
}

#[derive(Debug)]
pub struct ExprBuilder<F, OP1, OP2>
where
    F: ComputValue,
    OP1: Operator,
    OP2: Operator,
{
    /// The map contains expression trees with references.
    pub(super) id_to_node: RefCell<BTreeMap<Ident, Node<F, OP1, OP2>>>,
    pub(super) id_to_name: RefCell<BTreeMap<VariableNameId, String>>,
    pub(super) name_set: RefCell<HashSet<String>>,
}

impl<'a, F, OP1, OP2> ExprBuilder<F, OP1, OP2>
where
    F: ComputValue,
    OP1: Operator,
    OP2: Operator,
{
    pub fn new() -> ExprBuilder<F, OP1, OP2> {
        ExprBuilder {
            id_to_node: RefCell::new(BTreeMap::new()),
            id_to_name: RefCell::new(BTreeMap::new()),
            name_set: RefCell::new(HashSet::new()),
        }
    }

    pub fn new_variable(&'a self, name: &str) -> Expr<'a, F, OP1, OP2> {
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

    pub fn register_node_get_expr(&'a self, node: Node<F, OP1, OP2>) -> Expr<'a, F, OP1, OP2> {
        let ident = self.register_node(node);
        Expr { ident, eb: self }
    }

    /// register is a mutable operation on self.map. `register` is not explicitly mut, to allow Copy and
    /// ergonomic arithmetic syntax.
    fn register_node(&self, node: Node<F, OP1, OP2>) -> Ident {
        let ident = self.new_ident();
        let mut map = self.id_to_node.borrow_mut();
        map.insert(ident, node);
        ident
    }

    fn new_ident(&self) -> Ident {
        Ident(self.id_to_node.borrow().len())
    }
}

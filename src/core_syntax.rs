//! Definition of core syntax. The core syntax allows to use expressions like `x + y * z`,
//! and build a computation graph out of those expressions. The core syntax is generic and does not impose
//! type of variables underlying computation (like f32 vs f64) or what operations are actually implemented (like addition, or logarithm).
use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet};
use std::fmt::{self, Display};
use std::ops;

/// A type of the computed value (like, f32). [ops::Add] is needed so we can update the adjoins. [ops::Mul<f32>]
/// is to update parameters w.r.t. learning rate.
pub trait ComputValue<V = f32>:
    Clone
    + fmt::Display
    + fmt::Debug
    + DefaultAdjoin
    + ops::Mul<V, Output = Self>
    + ops::Add<Self, Output = Self>
{
}

/// Returns an initial adjoin for a type (a "1").
pub trait DefaultAdjoin {
    /// Return default value of adjoin ("1"). For simple type like f32, then the return value
    /// is obviously 1.0. But, when the user uses some dynamically-sized matrix, then it's not
    /// obvious what the size should be, and the input `value` should help with figuring those
    /// dynamic aspects of the type.
    fn default_adjoin(value: Self) -> Self;
}

/// Identifier of an [Expr][Expr]. Ident is [Copy] so we can have ergonomic syntax of building
/// the expression tree, like `y = a + b`.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
pub struct Ident(usize);

impl Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_{}", self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct NameId(Ident);

impl Display for NameId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "name({})", self.0)
    }
}

impl From<Ident> for NameId {
    fn from(value: Ident) -> Self {
        NameId(value)
    }
}

impl From<NameId> for Ident {
    fn from(value: NameId) -> Self {
        value.0
    }
}

impl<'a> From<&'a NameId> for &'a Ident {
    fn from(value: &'a NameId) -> Self {
        &value.0
    }
}

pub trait Operator: Clone + Copy + fmt::Debug + fmt::Display {}

/// A node in the expression tree built by the user. The node is Copy to allow ergonomic syntax (i.e. like, adding the nodes).
#[derive(Clone, Copy, Debug)]
pub enum ExprNode<F, OP1, OP2>
where
    F: ComputValue,
    OP1: Operator,
    OP2: Operator,
{
    /// A constant value.
    Const(F),
    /// A named variable.
    Variable(NameId),
    /// An unnamed model parameter with some initial value.
    Parameter(Option<NameId>, F),
    /// Arity-1 operation, like ln or sin.
    Ary1(OP1, Ident),
    /// Arity-2 operation, like addition.
    Ary2(OP2, Ident, Ident),
}

/// An identifier coupled with a reference to ExprBuilder, so it can be later used in further arithmetic operations.
/// Expr should be Copy so we can have ergonomic expressions like `y = v1 + v2` without additional `&` or `.clone()`.
#[derive(Clone, Debug)]
pub struct Expr<'a, F, OP1, OP2>
where
    F: ComputValue,
    OP1: Operator,
    OP2: Operator,
{
    pub ident: Ident, // TODO make ident private? or remove ident() method?

    // `eb` cannot be mut because mut is not Clone, therefore is not Copy, and we want Copy to be able
    // to do `a + b` on those expressions.
    // `eb` is pub so it's possible to create latent parameters.
    pub eb: &'a ExprBuilder<F, OP1, OP2>,
}

/// When I used `Copy` with `derive`, `ExprDMatrix` was not Copy :( Need to figure out why. Possibly related to
/// the differences [mentioned in the doc](https://doc.rust-lang.org/std/marker/trait.Copy.html#how-can-i-implement-copy).
impl<'a, F, OP1, OP2> Copy for Expr<'a, F, OP1, OP2>
where
    F: ComputValue,
    OP1: Operator,
    OP2: Operator,
{
}

impl<'a, F, OP1, OP2> Expr<'a, F, OP1, OP2>
where
    F: ComputValue,
    OP1: Operator,
    OP2: Operator,
{
    pub fn register_and_continue_expr(&self, node: ExprNode<F, OP1, OP2>) -> Expr<'a, F, OP1, OP2> {
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
    fn fmt_node(&self, f: &mut fmt::Formatter<'_>, node_ident: &Ident) -> fmt::Result {
        let id_to_node = self.eb.id_to_node.borrow();
        let node = id_to_node.get(node_ident).ok_or(fmt::Error)?;
        match node {
            ExprNode::Const(value) => write!(f, "{}", value)?,
            ExprNode::Parameter(name_id, _) => {
                let name = name_id
                    .and_then(|id| self.eb.get_name(&id))
                    .unwrap_or(format!("{}", node_ident));
                write!(f, "{}", name,)?;
            }
            ExprNode::Variable(name_id) => {
                let name = self.eb.get_name(name_id).unwrap();
                write!(f, "{}", name)?;
            }
            ExprNode::Ary1(op, ident) => {
                write!(f, "{}(", op)?;
                self.fmt_node(f, ident)?;
                write!(f, ")",)?;
            }
            ExprNode::Ary2(op, ident1, ident2) => {
                write!(f, "(")?;
                self.fmt_node(f, ident1)?;
                write!(f, "{}", op)?;
                self.fmt_node(f, ident2)?;
                write!(f, ")")?;
            }
        };
        Ok(())
    }
}

impl<'a, F, OP1, OP2> fmt::Display for Expr<'a, F, OP1, OP2>
where
    F: ComputValue,
    OP1: Operator,
    OP2: Operator,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_node(f, &self.ident)
    }
}

/// Expression builder holds state of the syntax tree, at the time when the user builds the
/// expressions. For example, `y=a*x+b` will hold y, a, x and b and relations between them.
/// Expression builder does not know how to calculate any of those, it just manages the
/// syntax tree.
///
/// All the methods, even are `&self` and not `&mut self` so it's possible to have an ergonomic
/// syntax when using individual expressions like `y=a+b` and not `y=&a + &b` or `y=a.clone() + b.clone()`.
#[derive(Debug)]
pub struct ExprBuilder<F, OP1, OP2>
where
    F: ComputValue,
    OP1: Operator,
    OP2: Operator,
{
    /// The map contains expression trees with references.
    pub(super) id_to_node: RefCell<BTreeMap<Ident, ExprNode<F, OP1, OP2>>>,
    id_to_name: RefCell<BTreeMap<NameId, String>>,
    name_set: RefCell<HashSet<String>>,
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
        let name_id = self.register_name(&ident, name).unwrap();
        let node = ExprNode::Variable(name_id);
        let mut id_to_node = self.id_to_node.borrow_mut();
        id_to_node.insert(ident, node);
        Expr { eb: &self, ident }
    }

    /// Create a new parameter without name, but with initial value.
    /// The parameter value is only used later when initializing [ComputeGraph].
    /// `new_parameter` is useful for operations that introduce latent parameters.
    pub fn new_parameter(&'a self, value: F) -> Expr<'a, F, OP1, OP2> {
        self.new_parameter_internal(None, value)
    }

    pub fn new_named_parameter(&'a self, name: &str, value: F) -> Expr<'a, F, OP1, OP2> {
        self.new_parameter_internal(Some(name), value)
    }

    fn new_parameter_internal(&'a self, name: Option<&str>, value: F) -> Expr<'a, F, OP1, OP2> {
        let ident = self.new_ident();
        let name_id = name.map(|name| self.register_name(&ident, name).unwrap());
        let node = ExprNode::Parameter(name_id, value);
        let mut id_to_node = self.id_to_node.borrow_mut();
        id_to_node.insert(ident, node);
        Expr { eb: &self, ident }
    }

    pub fn register_node_get_expr(&'a self, node: ExprNode<F, OP1, OP2>) -> Expr<'a, F, OP1, OP2> {
        let ident = self.register_node(node);
        Expr { ident, eb: self }
    }

    pub fn get_name(&self, name_id: &NameId) -> Option<String> {
        let id_to_name = self.id_to_name.borrow();
        id_to_name.get(name_id).map(|s| s.to_owned())
    }

    fn register_name(&self, ident: &Ident, name: &str) -> Result<NameId, String> {
        let name_id: NameId = ident.clone().into();

        let mut id_to_name = self.id_to_name.borrow_mut();
        if let Some(old_name) = id_to_name.insert(name_id, name.to_owned()) {
            return Err(format!(
                "Same {:?} registered twice, old name {:?}!",
                name_id, old_name
            ));
        }

        let mut name_set = self.name_set.borrow_mut();
        if !name_set.insert(name.to_owned()) {
            return Err(format!("Variable with name {} already exists", name));
        }
        Ok(name_id)
    }

    /// register is a mutable operation on self.map. `register` is not explicitly mut, to allow Copy and
    /// ergonomic arithmetic syntax.
    fn register_node(&self, node: ExprNode<F, OP1, OP2>) -> Ident {
        let ident = self.new_ident();
        let mut map = self.id_to_node.borrow_mut();
        map.insert(ident, node);
        ident
    }

    fn new_ident(&self) -> Ident {
        Ident(self.id_to_node.borrow().len())
    }
}

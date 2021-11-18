use egg::*;
use std::fmt::{self, Formatter, Display};


#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Lambda {
    Var(i32), // db index ($i)
    IVar(i32), // db index used by inventions (#i)
    App([Id; 2]), // f, x
    Lam([Id; 1]), // body
    Prim(egg::Symbol), // fallback, parses prims
    Programs(Vec<Id>),
}

type RecExpr = egg::RecExpr<Lambda>;

impl Display for Lambda {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Var(i) => write!(f, "${}", i),
            Self::IVar(i) => write!(f, "#{}", i),
            Self::App(_) => write!(f,"app"),
            Self::Lam(_) => write!(f,"lam"),
            Self::Prim(p) => write!(f,"{}",p),
            Self::Programs(_) => write!(f,"programs"),
        }
    }
}

impl Language for Lambda {
    fn matches(&self, other: &Self) -> bool {
        // consider only operator, not children. I believe (?) we do want to consider number of children based on the macro code.
        match (self,other) {
            (Self::Var(i), Self::Var(j)) => i == j,
            (Self::IVar(i), Self::IVar(j)) => i == j,
            (Self::App(_), Self::App(_)) => true,
            (Self::Lam(_), Self::Lam(_)) => true,
            (Self::Prim(p1), Self::Prim(p2)) => p1 == p2,
            (Self::Programs(p1), Self::Programs(p2)) => p1.len() == p2.len(),
            (_,_) => false,
        }
    }

    fn children(&self) -> &[Id] {
        match self {
            Self::Lam(ids) => ids,
            Self::App(ids) => ids,
            Self::Programs(ids) => ids,
            _ => &[],
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            Self::Lam(ids) => ids,
            Self::App(ids) => ids,
            Self::Programs(ids) => ids,
            _ => &mut [],
        }
    }

    fn display_op(&self) -> &dyn Display {
        unimplemented!("Use show(recexpr) to display a recexpr. This is because egg 0.6.0 hasnt fixed issue #83 so displaying things like $5 is not valid")
    }

    fn from_op_str(op_str: &str, children: Vec<Id>) -> Result<Self, String> {
        match op_str {
            "app" => {
                if children.len() != 2 {
                    return Err(format!("app needs 2 children, got {}", children.len()));
                }
                Ok(Self::App([children[0], children[1]]))
            },
            "lam" => {
                if children.len() != 1 {
                    return Err(format!("lam needs 1 child, got {}", children.len()));
                }
                Ok(Self::Lam([children[0]]))
            }
            "programs" => Ok(Self::Programs(children)),
            _ => {
                if children.len() != 0 {
                    return Err(format!("{} needs 0 children, got {}", op_str, children.len()))
                }
                if op_str.starts_with("$") {
                    let i = op_str.chars().skip(1).collect::<String>().parse::<i32>().unwrap();
                    Ok(Self::Var(i))
                } else if op_str.starts_with("#") {
                    let i = op_str.chars().skip(1).collect::<String>().parse::<i32>().unwrap();
                    Ok(Self::IVar(i))
                } else {
                    Ok(Self::Prim(egg::Symbol::from(op_str)))
                }
            },
        }
    }
}

pub fn show(e: &RecExpr) -> String {
    Expr::from(e.clone()).to_string()
}

pub struct Expr {
    pub nodes: Vec<Lambda>, // just like in a RecExpr but public
}

impl From<RecExpr> for Expr {
    fn from(e: RecExpr) -> Self {
        // todo you could (and should) actually grab it recursively, this is just some unsafe cheating during experimenting
        unsafe{ std::mem::transmute(e) }
    }
}

impl Expr {
    pub fn new(nodes: Vec<Lambda>) -> Self {
        Self { nodes: nodes}
    }
    pub fn var(i: i32) -> Self {
        Self { nodes: vec![Lambda::Var(i)] }
    }
    pub fn ivar(i: i32) -> Self {
        Self { nodes: vec![Lambda::IVar(i)] }
    }
    pub fn prim(p: egg::Symbol) -> Self {
        Self { nodes: vec![Lambda::Prim(p)] }
    }
    pub fn app(f: &Expr, x: &Expr) -> Self {
        let mut nodes = f.nodes.clone();
        nodes.extend(x.shifted_nodes(f.nodes.len()));
        let f_id = Id::from(f.nodes.len()-1);
        let x_id = Id::from(f.nodes.len() + x.nodes.len() - 1);
        nodes.push(Lambda::App([f_id, x_id]));
        Self::new(nodes)
    }
    pub fn lam(b: &Expr) -> Self{
        let mut nodes = b.nodes.clone();
        let b_id = Id::from(b.nodes.len()-1);
        nodes.push(Lambda::Lam([b_id]));
        Self::new(nodes)
    }
    pub fn programs(programs: &[Expr]) -> Self {
        let mut nodes = vec![];
        let mut root_ids = vec![];
        for p in programs {
            nodes.extend(p.shifted_nodes(nodes.len()));
            root_ids.push(Id::from(nodes.len() - 1));
        }
        nodes.push(Lambda::Programs(root_ids));
        Self::new(nodes)
    }

    fn shifted_nodes(&self, shift: usize) -> Vec<Lambda> {
        self.nodes.iter().cloned().map(|node|
            node.map_children(|id| Id::from(usize::from(id) + shift))
        ).collect()
    }

    fn write_child(&self, child:Id, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let node = &self.nodes[usize::from(child)];
        if node.is_leaf() {
            write!(f, "{}", node)?;
        } else {
            write!(f,"(")?;
            write!(f, "{}", node)?;
            for child in node.children() {
                write!(f," ")?;
                self.write_child(*child,f)?;
            };
            write!(f,")")?;
        }
        Ok(())
    }
}


impl Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.nodes.is_empty() {
            write!(f, "()")
        } else {
            self.write_child(Id::from(self.nodes.len()-1), f)
        }
    }
}
// impl std::ops::Index<Id> for Expr {
//     type Output = Lambda;
//     fn index(&self, id: Id) -> &Self::Output {
//         self.nodes[usize::from(id)]
//             .as_ref()
//             .unwrap_or_else(|| panic!("Invalid id {}", id))
//     }
// }

// impl std::ops::IndexMut<Id> for Expr {
//     fn index_mut(&mut self, id: Id) -> &mut Self::Output {
//         let id = self.find(id);
//         self.nodes[usize::from(id)]
//             .as_mut()
//             .unwrap_or_else(|| panic!("Invalid id {}", id))
//     }
// }
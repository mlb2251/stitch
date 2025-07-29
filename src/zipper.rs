use lambdas::ExprOwned;
pub use lambdas::{ZNode, ZId, LabelledZId};

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Zipper(Vec<ZNode>);


impl Zipper {

    #[inline(always)]
    pub fn ends_with_func(&self) -> bool {
        matches!(self.0.last(), Some(ZNode::Func))
    }

    #[inline(always)]
    pub fn function_arity(&self) -> usize {
        self.0.iter().rev().take_while(|znode| **znode == ZNode::Func).count()
    }

    #[inline(always)]
    pub fn depth_root_to_arg(&self) -> usize {
        self.0.iter().filter(|x| **x == ZNode::Body).count()
    }

    #[inline(always)]
    pub fn starts_with(&self, other: &Zipper) -> bool {
        self.0.starts_with(&other.0)
    }

    #[inline(always)]
    pub fn add_to_front(&mut self, node: ZNode) {
        self.0.insert(0, node);
    }

    #[inline(always)]
    pub fn add_to_end(&mut self, node: ZNode) {
        self.0.push(node);
    }

    #[inline(always)]
    pub fn remove_from_end(&mut self) {
        self.0.pop();
    }

    #[inline(always)]
    pub fn zip(&self, expr: &ExprOwned) -> ZId {
        expr.immut().zip(&self.0).idx
    }

}
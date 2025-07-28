use std::ops::{self, RangeFull};

pub use lambdas::{ZNode, ZId, LabelledZId};

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Zipper(Vec<ZNode>);

impl ops::Index<RangeFull> for Zipper
where
{
    type Output = [ZNode];

    fn index(&self, index: RangeFull) -> &Self::Output {
        &self.0[index]
    }
}

impl Zipper {
    
    pub fn ends_with_func(&self) -> bool {
        matches!(self.0.last(), Some(ZNode::Func))
    }

    pub fn function_arity(&self) -> usize {
        self.0.iter().rev().take_while(|znode| **znode == ZNode::Func).count()
    }

    pub fn depth_root_to_arg(&self) -> usize {
        self.0.iter().filter(|x| **x == ZNode::Body).count()
    }

    pub fn starts_with(&self, other: &Zipper) -> bool {
        self.0.starts_with(&other.0)
    }

    pub fn add_to_front(&mut self, node: ZNode) {
        self.0.insert(0, node);
    }

    pub fn add_to_end(&mut self, node: ZNode) {
        self.0.push(node);
    }

    pub fn remove_from_end(&mut self) {
        self.0.pop();
    }
}
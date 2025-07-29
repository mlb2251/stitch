use lambdas::{AnalyzedExpr, ExprOwned, ExprSet, FreeVarAnalysis, Idx};
pub use lambdas::{ZNode, ZId, LabelledZId};
use rustc_hash::FxHashMap;

use crate::{insert_arg_ivars, Arg, Cost, ExpandsTo, ZIdExtension};

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

#[derive(Clone, Debug, Default)]
pub struct Zippers {
    pub zid_of_zip: FxHashMap<Zipper, ZId>,
    pub zip_of_zid: Vec<Zipper>,
    pub arg_of_zid_node: Vec<FxHashMap<Idx,Arg>>,
}

impl Zippers {

    pub fn get_interned_idx(&self, zipper: &Zipper) -> Option<ZId> {
        self.zid_of_zip.get(zipper).cloned()
    }

    pub fn add_empty(&mut self, empty_zid: ZId) {
        self.zid_of_zip.insert(Zipper::default(), empty_zid);
        self.zip_of_zid.push(Zipper::default());
        self.arg_of_zid_node.push(FxHashMap::default());
    }

    pub fn add_arg(&mut self, zid: ZId, node: Idx, cost: Cost, expands_to: ExpandsTo) { 
        self.arg_of_zid_node[zid].insert(node,
            Arg { shifted_id: node, unshifted_id: node, shift: 0, cost, expands_to });

    }

    pub fn extend_zipper(&mut self, unextended_zid: ZId, extended_node: Idx, unextended_node: Idx, znode: ZNode) -> usize {
        let mut zip = self.zip_of_zid[unextended_zid].clone();
        zip.add_to_front(znode);
        let zip_of_zid = &mut self.zip_of_zid;
        let arg_of_zid_node = &mut self.arg_of_zid_node;
        let zid = self.zid_of_zip.entry(zip.clone()).or_insert_with(|| {
            let zid = zip_of_zid.len();
            zip_of_zid.push(zip);
            arg_of_zid_node.push(FxHashMap::default());
            zid
        });
        // add new zid to this node
        // give it the same arg
        let arg = self.arg_of_zid_node[unextended_zid][&unextended_node].clone();
        self.arg_of_zid_node[*zid].insert(extended_node, arg);
        *zid
    }

    pub fn handle_shift(&mut self, extended_zid: ZId, unextended_zid: ZId, extended_node: Idx, unextended_node: Idx, analyzed_free_vars: &mut AnalyzedExpr<FreeVarAnalysis>, set: &mut ExprSet) {
        let zip = &self.zip_of_zid[extended_zid];
        let mut arg: Arg = self.arg_of_zid_node[unextended_zid][&unextended_node].clone();
        // shift the arg but keep the unshifted part the same
        if !analyzed_free_vars.analyze_get(set.get(arg.shifted_id)).is_empty() {
            // the arg has free vars so we should actually downshift it by 1
            if analyzed_free_vars[arg.shifted_id].contains(&0) {
                // furthermore one of those vars is a 0 then it will get shifted to -1, so we handle that slightly specially
                // by inserting an IVar to indicate this

                // how many lambdas are along this zipper? (including most recent one)
                let depth_root_to_arg = zip.depth_root_to_arg() as i32;

                // find all pointers to $0 (this is the `init_depth` parameter) and replace then with #(num_lams - 1) that is
                // point past all lambdas except the newly added one. For example if there were no lambdas other than the
                // newly added one this would be num_lams=1 so it'd be #0.
                arg.shifted_id = insert_arg_ivars(&mut set.get_mut(arg.shifted_id), depth_root_to_arg-1, 0, analyzed_free_vars);
            }
            arg.shifted_id = set.get_mut(arg.shifted_id).shift(-1, 0, analyzed_free_vars);
            arg.shift -= 1;
        }
        self.arg_of_zid_node[extended_zid].insert(extended_node, arg);
    }

    pub fn compute_extensions(&self) -> Vec<ZIdExtension> {
        self.zip_of_zid.iter().map(|zip| {
            let mut zip_body = zip.clone();
            zip_body.add_to_end(ZNode::Body);
            let mut zip_arg = zip.clone();
            zip_arg.add_to_end(ZNode::Arg);
            let mut zip_func = zip.clone();
            zip_func.add_to_end(ZNode::Func);
            ZIdExtension {
                body: self.zid_of_zip.get(&zip_body).copied(),
                arg: self.zid_of_zip.get(&zip_arg).copied(),
                func: self.zid_of_zip.get(&zip_func).copied(),
            }
        }).collect()
    }

    pub fn print_stats(&self) {
        println!("{} zips", self.zip_of_zid.len());
        println!("arg_of_zid_node size: {}", self.arg_of_zid_node.len())
    }

}
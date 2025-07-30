use lambdas::{AnalyzedExpr, Expr, ExprOwned, ExprSet, FreeVarAnalysis, Idx};
pub use lambdas::{ZNode, ZId, LabelledZId};
use rustc_hash::FxHashMap;

use crate::{insert_arg_ivars, Arg, Cost, ExpandsTo, ZIdExtension, EMPTY_ZID};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Zipper {
    EmptyZipper,
    ConsZipper {
        head: ZNode,
        tail: ZId,
    },
}


impl Zipper {

    #[inline(always)]
    pub fn ends_with_func(&self, zippers: &Zippers) -> bool {
        return self.function_arity(zippers) > 0
    }

    #[inline(always)]
    pub fn function_arity(&self, zippers: &Zippers) -> usize {
        let mut ptr = self;
        let mut count = 0;
        // looking at the last run of Funcs in the zipper
        while let Zipper::ConsZipper { head, tail } = ptr {
            match head {
                ZNode::Func => count += 1,
                _ => count = 0
            }
            ptr = &zippers.zip_of_zid[*tail];
        }
        count
    }

    #[inline(always)]
    pub fn depth_root_to_arg(&self, zippers: &Zippers) -> usize {
        // looking at the first run of Body in the zipper
        let mut ptr = self;
        let mut count = 0;
        while let Zipper::ConsZipper { head: ZNode::Body, tail } = ptr {
            count += 1;
            ptr = &zippers.zip_of_zid[*tail];
        }
        count
    }

    #[inline(always)]
    pub fn starts_with(&self, other: &Zipper, zippers: &Zippers) -> bool {
        // self.0.starts_with(&other.0)
        let mut self_ptr = self;
        let mut other_ptr = other;
        while let (Zipper::ConsZipper { head: self_head, tail: self_tail }, Zipper::ConsZipper { head: other_head, tail: other_tail }) = (self_ptr, other_ptr) {
            if self_head != other_head {
                return false;
            }
            self_ptr = &zippers.zip_of_zid[*self_tail];
            other_ptr = &zippers.zip_of_zid[*other_tail];
        }
        // at this point, at least one of the pointers is EmptyZipper, it should be the other one for self to start with the other
        other_ptr == &Zipper::EmptyZipper
    }

    #[inline(always)]
    pub fn add_to_end(&mut self, node: ZNode, zippers: &mut Zippers) -> ZId {
        match self {
            Zipper::EmptyZipper => {
                *self = Zipper::ConsZipper { head: node, tail: EMPTY_ZID };
            },
            Zipper::ConsZipper { head, tail } => {
                let mut tail_zip = zippers.zip_of_zid[*tail].clone();
                *self = Zipper::ConsZipper { head: head.clone(), tail: tail_zip.add_to_end(node, zippers) };
            }
        }
        zippers.add_zip(self.clone())
    }

    #[inline(always)]
    pub fn remove_from_end(&mut self, zippers: &mut Zippers) -> ZId {
        match self {
            Zipper::EmptyZipper => {},
            Zipper::ConsZipper { head, tail } => {
                if *tail == EMPTY_ZID {
                    *self = Zipper::EmptyZipper;
                } else {
                    let mut tail_zip = zippers.zip_of_zid[*tail].clone();
                    *self = Zipper::ConsZipper { head: head.clone(), tail: tail_zip.remove_from_end(zippers) };
                }
            }
        }
        zippers.add_zip(self.clone())
    }

    #[inline(always)]
    pub fn zip(&self, expr: &ExprOwned, zippers: &Zippers) -> ZId {
        
        self._zip(expr.immut(), zippers).idx
    }

    pub fn _zip<'a>(&self, expr: Expr<'a>, zippers: &Zippers) -> Expr<'a> {
        match self {
            Zipper::EmptyZipper => expr,
            Zipper::ConsZipper { head, tail } => {
                // let head_vec = vec![head.clone()];
                let expr = expr.zip_once(head.clone());
                let expr = zippers.zip_of_zid[*tail]._zip(expr, zippers);
                expr
            }
        }
    }

}

impl Default for Zipper {
    fn default() -> Self {
        Zipper::EmptyZipper
    }
}

#[derive(Clone, Debug, Default)]
pub struct Zippers {
    pub zid_of_zip: FxHashMap<Zipper, ZId>,
    pub zip_of_zid: Vec<Zipper>,
    pub arg_of_zid_node: Vec<FxHashMap<Idx,Arg>>,
}

impl Zippers {

    #[inline(always)]
    pub fn get_interned_idx(&self, zipper: &Zipper) -> Option<ZId> {
        self.zid_of_zip.get(zipper).cloned()
    }

    #[inline(always)]
    pub fn add_empty(&mut self, empty_zid: ZId) {
        self.zid_of_zip.insert(Zipper::default(), empty_zid);
        self.zip_of_zid.push(Zipper::default());
        self.arg_of_zid_node.push(FxHashMap::default());
    }

    #[inline(always)]
    pub fn add_arg(&mut self, zid: ZId, node: Idx, cost: Cost, expands_to: ExpandsTo) { 
        self.arg_of_zid_node[zid].insert(node,
            Arg { shifted_id: node, unshifted_id: node, shift: 0, cost, expands_to });

    }

    #[inline(always)]
    pub fn extend_zipper(&mut self, unextended_zid: ZId, extended_node: Idx, unextended_node: Idx, znode: ZNode) -> usize {
        let zip = Zipper::ConsZipper {
            head: znode,
            tail: unextended_zid,
        };
        let zid = self.add_zip(zip);
        // add new zid to this node
        // give it the same arg
        let arg = self.arg_of_zid_node[unextended_zid][&unextended_node].clone();
        self.arg_of_zid_node[zid].insert(extended_node, arg);
        zid
    }

    #[inline(always)]
    fn add_zip(&mut self, zip: Zipper) -> usize {
        let zip_of_zid = &mut self.zip_of_zid;
        let arg_of_zid_node = &mut self.arg_of_zid_node;
        let zid = self.zid_of_zip.entry(zip.clone()).or_insert_with(|| {
            let zid = zip_of_zid.len();
            zip_of_zid.push(zip);
            arg_of_zid_node.push(FxHashMap::default());
            zid
        });
        *zid
    }
    
    #[inline(always)]
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
                let depth_root_to_arg = zip.depth_root_to_arg(self) as i32;

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

    #[inline(always)]
    pub fn compute_extensions(&mut self) -> Vec<ZIdExtension> {
        let mut extensions = vec![];
        for zip in self.zip_of_zid.clone().into_iter() {
            let mut zip_body = zip.clone();
            zip_body.add_to_end(ZNode::Body, self);
            let mut zip_arg = zip.clone();
            zip_arg.add_to_end(ZNode::Arg, self);
            let mut zip_func = zip.clone();
            zip_func.add_to_end(ZNode::Func, self);
            extensions.push(ZIdExtension {
                body: self.zid_of_zip.get(&zip_body).copied(),
                arg: self.zid_of_zip.get(&zip_arg).copied(),
                func: self.zid_of_zip.get(&zip_func).copied(),
            });
        }
        extensions
    }

    pub fn print_stats(&self) {
        println!("{} zips", self.zip_of_zid.len());
        println!("arg_of_zid_node size: {}", self.arg_of_zid_node.len())
    }

}
use lambdas::*;
use rustc_hash::{FxHashSet};

pub fn topological_ordering(root: Idx, set: &ExprSet) -> Vec<Idx> {
    let mut vec = Vec::new();
    topological_ordering_rec(root, set, &mut vec);
    // let alt = (0..=root).collect::<Vec<usize>>().into_iter().map(|x| Idx::from(x)).collect::<Vec<Idx>>();
    // assert_eq!(vec, alt);
    vec
}

fn topological_ordering_rec(root: Idx, set: &ExprSet, vec: &mut Vec<Idx>) {
    for child in set.get(root).children(){
        topological_ordering_rec(child, set, vec);
    }
    if !vec.contains(&root) {
        // if we're already a child of someone else earlier we dont need to be readded
        vec.push(root);
    }
}

//#[inline(never)]
pub fn associate_tasks(roots: &[Idx], set: &ExprSet, corpus_span: &Span, task_of_root_idx: &[usize]) -> Vec<FxHashSet<usize>> {

    // this is the map from egraph node ids to tasks (represented with unique usizes) that we will be building
    let mut tasks_of_node = vec![FxHashSet::default(); corpus_span.len()];

    assert_eq!(roots.len(), task_of_root_idx.len());

    // since the tasks may not be listed in any specific order, we need to keep track of whether we've already
    // made an Idx for a given task or not
    for (root, task) in roots.iter().zip(task_of_root_idx) {
        associate_task_rec(*root, set, *task, &mut tasks_of_node)
    }

    // defensive sanity check that each entry is non-empty
    assert!(tasks_of_node.iter().all(|tasks| !tasks.is_empty()));

    tasks_of_node
}

fn associate_task_rec(node: Idx, set: &ExprSet, task_id: usize, tasks_of_node: &mut Vec<FxHashSet<usize>>) {
    // tasks_of_node.entry(node).or_default().insert(task_id);
    tasks_of_node[node].insert(task_id);
    for child in set.get(node).children() {
        associate_task_rec(child, set, task_id, tasks_of_node);
    }
}

#[inline]
pub fn insert_arg_ivars(e: &mut ExprMut, set_to: i32, init_depth: i32, analyzed_free_vars: &mut AnalyzedExpr<FreeVarAnalysis>) -> Idx {
    analyzed_free_vars.analyze_to(e.set, e.idx);
    if analyzed_free_vars[e.idx].is_empty()
        || *analyzed_free_vars[e.idx].iter().max().unwrap() < init_depth
    {
        return e.idx; // no free vars
    }

    match e.node().clone() {
        Node::Prim(_) => e.idx,
        Node::Var(i) => if i == init_depth { e.set.add(Node::IVar(set_to)) } else { e.idx },
        Node::IVar(_) => e.idx,
        Node::App(f, x) => {
            let f = insert_arg_ivars(&mut e.get(f), set_to, init_depth, analyzed_free_vars);
            let x = insert_arg_ivars(&mut e.get(x), set_to, init_depth, analyzed_free_vars);
            e.set.add(Node::App(f, x))
        },
        Node::Lam(b) => {
            let b = insert_arg_ivars(&mut e.get(b), set_to, init_depth + 1, analyzed_free_vars);
            e.set.add(Node::Lam(b))
        },
    }
}
use egg::*;
// use std::collections::HashMap;

// pub fn var(s: &str) -> Var {
//     s.parse().unwrap()
// }

/// finds everywhere the rewrite rules matches and applies it to each of them
/// and rebuilds the egraph. Will only apply to matches that are visible before
/// any rewriting occurs. This is the same as running a runner with an iter limit of 1.
/// I guess I'm not using this in the code right now bc I like the runner's report.
// pub fn apply_everywhere_once<L,A>(rules_: &[&str], egraph: &mut EGraph<L,A>)
// where 
//     L: Language,
//     A: Analysis<L>
// {
//     let rules: Vec<Rewrite<L,A>> = rules(rules_);
//     let matches: Vec<Vec<SearchMatches>> = rules.iter().map(|r| r.search(egraph)).collect();
//     for (r,m) in rules.iter().zip(matches) {
//         let hits = r.apply(egraph, &m).len();
//         println!("(applied {} {} times out of {} matches)",r.name(),hits, m.len());
//     }
//     egraph.rebuild();
// }

// pub fn saturate<L,A>(rules_: &[&str], render: bool, out_dir: String, egraph: EGraph<L,A>) -> EGraph<L,A>
// where 
//     L: Language,
//     A: Analysis<L>
// {
//     let rules: Vec<Rewrite<L,A>> = rules(rules_);
//     let mut runner = Runner::default()
//         .with_egraph(egraph)
//         .with_iter_limit(400)
//         .with_scheduler(SimpleScheduler)
//         .with_time_limit(core::time::Duration::from_secs(200))
//         .with_node_limit(3000000);
    
//     if render {
//         runner = runner.with_hook(
//         {
//             let out_dir = out_dir.clone(); // silly thing to clone into the closure
//             move |runner|{
//                 let iter = runner.iterations.len();
//                 println!("Iter {}: {}", iter, egraph_info(&runner.egraph));
//                 save(&runner.egraph, format!("3_propagate_{}",iter).as_str(), &out_dir);
//                 Ok(())
//             }
//         });
//     }

//     let runner = runner.run(rules.iter());
//     runner.print_report();
//     runner.egraph
// }

// pub fn run_pretty<L,A>(rule_: &str, name:&str, egraph: &mut EGraph<L,A>)
// where 
//     L: Language,
//     A: Analysis<L>
// {
//     let rule: Rewrite<L,A> = rule(rule_);
//     let matches = rule.search(egraph);
//     egraph.dot().to_png(format!("target/match_{}_0pre.png",name)).unwrap();
//     rule.apply(egraph, &matches).len();
//     egraph.dot().to_png(format!("target/match_{}_1post.png",name)).unwrap();
//     egraph.rebuild();
//     egraph.dot().to_png(format!("target/match_{}_2rebuild.png",name)).unwrap();
// }



// pub fn rule_map<L,A>() -> HashMap<String,Rewrite<L,A>> 
// where 
//     L: Language,
//     A: Analysis<L>
//     {
//     vec![
//     ].into_iter().map(|r:Rewrite<L,A>| (r.name().to_string(),r)).collect()
// }

// // ownership is a pain so this is a helper
// pub fn rule(name: &str) -> Rewrite<L,A> {
//     rule_map().remove(name).expect(format!("rule {} not found",name).as_str())
// }

// pub fn rules(names: &[&str]) -> Vec<Rewrite<L,A>> {
//     names.iter().map(|name|rule(name)).collect()
// }


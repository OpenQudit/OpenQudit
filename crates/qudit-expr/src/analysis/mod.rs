use egg::*;
use num::Signed;
use num::Zero;
use ordered_float::NotNan;

use crate::ComplexExpression;
use crate::Expression;

// #[cfg(test)]
// mod extract;

pub type EGraph = egg::EGraph<TrigLanguage, ConstantFold>;
pub type Constant = NotNan<f64>;

define_language! {
    pub enum TrigLanguage {
        "pi" = Pi,

        "~" = Neg([Id; 1]),
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),

        "pow" = Pow([Id; 2]),
        "sqrt" = Sqrt(Id),
        "sin" = Sin(Id),
        "cos" = Cos(Id),

        Constant(Constant),
        Symbol(Symbol),
    }
}

fn is_not_zero(var: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| {
        if let Some(n) = &egraph[subst[var]].data {
            *(n.0) != 0.0
        } else {
            false
        }
    }
}

fn is_non_negative_conservative(var: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| {
        if let Some(n) = &egraph[subst[var]].data {
            *(n.0) >= 0.0
        } else {
            false
        }
    }
}

#[allow(dead_code)]
fn all_not_zero(vars: &[&str]) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let vars: Vec<_> = vars.iter().map(|v| v.parse().unwrap()).collect();
    move |egraph, _, subst| {
        vars.iter().all(|&v| {
            if let Some(n) = &egraph[subst[v]].data {
                *(n.0) != 0.0
            } else {
                true
            }
        })
    }
}

fn cmp<T: PartialOrd>(a: &Option<T>, b: &Option<T>) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(a), Some(b)) => a.partial_cmp(b).unwrap(),
    }
}

use core::f64;
use std::cmp::Ordering;
use std::collections::HashMap;

#[allow(dead_code)]
struct SineExtractor<'a> {
    costs: HashMap<Id, (f64, TrigLanguage)>,
    egraph: &'a EGraph,
}

impl<'a> SineExtractor<'a> {
    pub fn new(egraph: &'a EGraph) -> Self {
        let costs = HashMap::default();
        let mut extractor = SineExtractor { costs, egraph };
        extractor.calculate_costs();
        extractor
    }

    pub fn extract_sine(&self, eclass: Id) -> Option<RecExpr<TrigLanguage>> {
        let id = self.egraph.find(eclass);
        let eclass = &self.egraph[id];

        // find best cosine node in eclass
        let mut best_cost = None;
        let mut best_node = None;

        for node in &eclass.nodes {
            if let TrigLanguage::Sin(id) = node {
                let cost = self.costs[&id].0;
                if best_cost.is_none() || cost < best_cost.unwrap() {
                    best_cost = Some(cost);
                    best_node = Some(node.clone());
                }
            }
        }

        if let Some(node) = best_node {
            let expr = node.build_recexpr(|id| self.find_best_node(id).clone());
            Some(expr)
        } else {
            None
        }
    }

    pub fn find_best_node(&self, eclass: Id) -> &TrigLanguage {
        &self.costs[&self.egraph.find(eclass)].1
    }

    fn calculate_costs(&mut self) {
        let mut did_something = true;
        while did_something {
            did_something = false;

            for class in self.egraph.classes() {
                let pass = self.make_pass(class);
                match (self.costs.get(&class.id), pass) {
                    (None, Some(new)) => {
                        self.costs.insert(class.id, new);
                        did_something = true;
                    }
                    (Some(old), Some(new)) if new.0 < old.0 => {
                        self.costs.insert(class.id, new);
                        did_something = true;
                    }
                    _ => {}
                }
            }
        }

        for class in self.egraph.classes() {
            if !self.costs.contains_key(&class.id) {
                println!("failed to calculate cost for {:?}", class);
            }
        }
    }

    fn make_pass(
        &mut self,
        eclass: &EClass<TrigLanguage, Option<(NotNan<f64>, RecExpr<ENodeOrVar<TrigLanguage>>)>>,
    ) -> Option<(f64, TrigLanguage)> {
        let (cost, node) = eclass
            .iter()
            .map(|n| (self.node_total_cost(n), n))
            .min_by(|a, b| cmp(&a.0, &b.0))
            .unwrap_or_else(|| panic!("eclass is empty"));
        cost.map(|c| (c, node.clone()))
    }

    fn node_total_cost(&mut self, enode: &TrigLanguage) -> Option<f64> {
        let eg = &self.egraph;
        let has_cost = |id| self.costs.contains_key(&eg.find(id));
        // TODO: these hashes can be quite expensive, see if we can cache
        // the enode.all(has_cost) cost in maybe a vector?
        if enode.all(has_cost) {
            Some(self.node_cost(enode))
        } else {
            None
        }
    }

    fn node_cost(&self, enode: &TrigLanguage) -> f64 {
        let op_cost = match enode {
            TrigLanguage::Constant(_) => 0.5,
            TrigLanguage::Neg(_) => 1.0,
            TrigLanguage::Add(_) | TrigLanguage::Sub(_) => 1.0,
            TrigLanguage::Mul(_) | TrigLanguage::Div(_) => 5.0,
            TrigLanguage::Pow(_)
            | TrigLanguage::Sqrt(_)
            | TrigLanguage::Sin(_)
            | TrigLanguage::Cos(_) => 50.0,
            _ => 0.0,
        };
        enode.fold(op_cost, |acc, id| acc + self.costs[&id].0)
    }
}

// struct TestExtractor<'a> {
//     memory: FxHashMap<Id, TrigLanguage>,
//     currently_processing: FxHashSet<Id>,
//     egraph: &'a EGraph,
// }

// impl<'a> TestExtractor<'a> {
//     pub fn new(e: &'a EGraph) -> Self {
//         let memory = FxHashMap::default();
//         let currently_processing = FxHashSet::default();
//         TestExtractor { memory, currently_processing, egraph: e }
//     }

//     pub fn extract_best(&mut self, eclass: Id) -> RecExpr<TrigLanguage> {
//         let root = self.egraph.find(eclass);
//         let node = self.calculate_best_node(root);
//         node.1.build_recexpr(|id| self.memory.get(&id).unwrap().clone())
//     }

//     pub fn calculate_best_node(&mut self, eclass: Id) -> (f64, TrigLanguage) {
//         let id = &self.egraph.find(eclass);
//         if self.currently_processing.contains(&id) {
//             return (f64::INFINITY, TrigLanguage::Constant(NotNan::new(f64::INFINITY).unwrap()));
//         }
//         if let Some(node) = self.memory.get(id) {
//             return (0.0, node.clone());
//         }

//         for node in &self.egraph[*id].nodes {
//             if node.is_leaf() {
//                 self.memory.insert(*id, node.clone());
//                 return (0.0, node.clone());
//             }
//         }

//         self.currently_processing.insert(*id);
//         let (cost, node) = self.egraph[*id].iter()
//             .map(|n| (self.evaluate_node(n), n)).min_by(|a, b| if a.0 < b.0 { Ordering::Less } else if a.0 == b.0 { Ordering::Equal } else { Ordering::Greater }).unwrap();
//         self.currently_processing.remove(id);
//         self.memory.insert(*id, node.clone());
//         (cost, node.clone())
//     }

//     pub fn evaluate_node(&mut self, enode: &'a TrigLanguage) -> f64 {
//         match enode {
//             TrigLanguage::Neg([a]) => self.calculate_best_node(*a).0 + 1.0,
//             TrigLanguage::Add([a, b]) => self.calculate_best_node(*a).0 + self.calculate_best_node(*b).0 + 1.0,
//             TrigLanguage::Sub([a, b]) => self.calculate_best_node(*a).0 + self.calculate_best_node(*b).0 + 1.0,
//             TrigLanguage::Mul([a, b]) => self.calculate_best_node(*a).0 + self.calculate_best_node(*b).0 + 5.0,
//             TrigLanguage::Div([a, b]) => self.calculate_best_node(*a).0 + self.calculate_best_node(*b).0 + 10.0,
//             TrigLanguage::Pow([a, b]) => self.calculate_best_node(*a).0 + self.calculate_best_node(*b).0 + 100.0,
//             TrigLanguage::Sqrt(a) => self.calculate_best_node(*a).0 + 50.0,
//             TrigLanguage::Sin(a) => self.calculate_best_node(*a).0 + 50.0,
//             TrigLanguage::Cos(a) => self.calculate_best_node(*a).0 + 50.0,
//             _ => panic!("unexpected node"),
//         }
//     }
// }

struct TrigExprExtractor<'a> {
    costs: Vec<(f64, TrigLanguage)>,
    egraph: &'a EGraph,
    has_changed: bool,
}

impl<'a> TrigExprExtractor<'a> {
    pub fn new(egraph: &'a EGraph) -> Self {
        // let costs = FxHashMap::default();
        let mut max_id = 0usize;
        for class in egraph.classes() {
            let id = unsafe { std::mem::transmute::<Id, u32>(class.id) } as usize;
            if id > max_id {
                max_id = id
            }
        }
        let costs = vec![(-1.0, TrigLanguage::Pi); max_id + 1];
        let mut extractor = TrigExprExtractor {
            costs,
            egraph,
            has_changed: false,
        };
        extractor.calculate_costs();
        extractor
    }

    pub fn extract_best(&mut self, eclass: Id) -> RecExpr<TrigLanguage> {
        let root = self.get_cost(self.egraph.find(eclass)).1.clone();
        let expr = root.build_recexpr(|id| self.extract_best_node(id));
        // TODO: This is creates a greedy search when simplifying many expressions
        // in a row: The expression's that are simplified later are effected greatly
        // by the extractions done earlier. There may be better extractions over
        // many expressions if a simultaneous extraction is done, but this is
        // not simple to implement. // Potential for signficant speed up here
        if self.has_changed {
            self.recalculate_costs();
            self.has_changed = false;
        }
        expr
    }

    pub fn extract_best_node(&mut self, eclass: Id) -> TrigLanguage {
        let id = &self.egraph.find(eclass);
        let (cost, enode) = self.get_cost(*id).clone();
        if cost != 0.0 {
            self.put_cost(*id, (cost, enode.clone()));
            self.has_changed = true;
        }
        enode
    }

    #[allow(dead_code)]
    pub fn find_best_node(&self, eclass: Id) -> &TrigLanguage {
        &self.get_cost(self.egraph.find(eclass)).1
    }

    fn recalculate_costs(&mut self) {
        let mut did_something = true;
        while did_something {
            did_something = false;

            for class in self.egraph.classes() {
                let pass = self.make_repass(class);
                let old = self.get_cost(class.id);
                match (old, pass) {
                    (old, new) if old.0 < 0.0 || new < old.0 => {
                        self.put_cost(class.id, (new, old.1.clone()));
                        did_something = true;
                    }
                    _ => {}
                }
            }
        }
    }

    fn calculate_costs(&mut self) {
        let mut did_something = true;
        while did_something {
            did_something = false;

            for class in self.egraph.classes() {
                let pass = self.make_pass(class);
                match (self.get_cost(class.id), pass) {
                    (old, Some(new)) => {
                        if old.0 < 0.0 {
                            self.put_cost(class.id, new);
                            did_something = true;
                        } else if new.0 > 0.0 && new.0 < old.0 {
                            self.put_cost(class.id, new);
                            did_something = true;
                        }
                    }
                    _ => {}
                }
            }
        }

        for class in self.egraph.classes() {
            if self.get_cost(class.id).0 < 0.0 {
                println!("failed to calculate cost for {:?}", class);
            }
        }
    }

    fn make_pass(
        &mut self,
        eclass: &EClass<TrigLanguage, Option<(NotNan<f64>, RecExpr<ENodeOrVar<TrigLanguage>>)>>,
    ) -> Option<(f64, TrigLanguage)> {
        let (cost, node) = eclass
            .iter()
            .map(|n| (self.node_total_cost(n), n))
            .min_by(|a, b| cmp(&a.0, &b.0))
            .unwrap_or_else(|| panic!("eclass is empty"));
        cost.map(|c| (c, node.clone()))
    }

    fn make_repass(
        &mut self,
        eclass: &EClass<TrigLanguage, Option<(NotNan<f64>, RecExpr<ENodeOrVar<TrigLanguage>>)>>,
    ) -> f64 {
        eclass
            .iter()
            .map(|n| self.node_cost(n))
            .min_by(|a, b| match a < b {
                true => Ordering::Less,
                false => Ordering::Greater,
            })
            .unwrap()
    }

    fn node_total_cost(&mut self, enode: &TrigLanguage) -> Option<f64> {
        let eg = &self.egraph;
        let has_cost = |id| self.get_cost(eg.find(id)).0 >= 0.0;
        if enode.all(has_cost) {
            Some(self.node_cost(enode))
        } else {
            None
        }
    }

    fn node_cost(&self, enode: &TrigLanguage) -> f64 {
        let op_cost = match enode {
            TrigLanguage::Constant(_) => 0.5,
            TrigLanguage::Neg(_) => 1.0,
            TrigLanguage::Add(_) | TrigLanguage::Sub(_) => 1.0,
            TrigLanguage::Mul(_) | TrigLanguage::Div(_) => 5.0,
            TrigLanguage::Sqrt(_) | TrigLanguage::Sin(_) | TrigLanguage::Cos(_) => 50.0,
            TrigLanguage::Pow(_) => 100.0,
            _ => 0.0,
        };
        enode.fold(op_cost, |acc, id| acc + self.get_cost(id).0)
    }

    #[inline(always)]
    fn get_cost(&self, id: Id) -> &(f64, TrigLanguage) {
        &self.costs[unsafe { std::mem::transmute::<Id, u32>(id) } as usize]
    }

    #[inline(always)]
    fn put_cost(&mut self, id: Id, cost: (f64, TrigLanguage)) {
        self.costs[unsafe { std::mem::transmute::<Id, u32>(id) } as usize] = cost;
    }
}

struct TrigCostFn;
impl CostFunction<TrigLanguage> for TrigCostFn {
    type Cost = f64;
    fn cost<C>(&mut self, enode: &TrigLanguage, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let op_cost = match enode {
            TrigLanguage::Constant(_) => 0.5,
            TrigLanguage::Neg(_) => 1.0,
            TrigLanguage::Add(_) | TrigLanguage::Sub(_) => 1.0,
            TrigLanguage::Mul(_) | TrigLanguage::Div(_) => 5.0,
            TrigLanguage::Pow(_)
            | TrigLanguage::Sqrt(_)
            | TrigLanguage::Sin(_)
            | TrigLanguage::Cos(_) => 50.0,
            _ => 0.0,
        };

        enode.fold(op_cost, |acc, id| acc + costs(id))
    }
}

pub fn can_multiply(a: NotNan<f64>, b: NotNan<f64>) -> bool {
    if !a.is_zero() && a.is_positive() && a < NotNan::new(1e-15).unwrap() {
        return false;
    }
    if !b.is_zero() && b.is_positive() && b < NotNan::new(1e-15).unwrap() {
        return false;
    }
    if !a.is_zero() && a.is_negative() && a > NotNan::new(-1e-15).unwrap() {
        return false;
    }
    if !b.is_zero() && b.is_negative() && b > NotNan::new(-1e-15).unwrap() {
        return false;
    }
    if a > NotNan::new(1e15).unwrap() || b > NotNan::new(1e15).unwrap() {
        return false;
    }
    if a < NotNan::new(-1e15).unwrap() || b < NotNan::new(-1e15).unwrap() {
        return false;
    }
    a.is_finite() && b.is_finite() && !a.is_subnormal() && !b.is_subnormal()
}

pub fn can_divide(a: NotNan<f64>, b: NotNan<f64>) -> bool {
    if !a.is_zero() && a.is_positive() && a < NotNan::new(1e-15).unwrap() {
        return false;
    }
    if !b.is_zero() && b.is_positive() && b < NotNan::new(1e-15).unwrap() {
        return false;
    }
    if !a.is_zero() && a.is_negative() && a > NotNan::new(-1e-15).unwrap() {
        return false;
    }
    if !b.is_zero() && b.is_negative() && b > NotNan::new(-1e-15).unwrap() {
        return false;
    }
    if a > NotNan::new(1e15).unwrap() || b > NotNan::new(1e15).unwrap() {
        return false;
    }
    if a < NotNan::new(-1e15).unwrap() || b < NotNan::new(-1e15).unwrap() {
        return false;
    }
    a.is_finite() && b.is_finite() && !b.is_zero() && !b.is_subnormal() && !a.is_subnormal()
}

#[derive(Default)]
pub struct ConstantFold;
impl Analysis<TrigLanguage> for ConstantFold {
    type Data = Option<(Constant, PatternAst<TrigLanguage>)>;

    fn make(egraph: &mut EGraph, enode: &TrigLanguage) -> Self::Data {
        let x = |i: &Id| egraph[*i].data.as_ref().map(|d| d.0);
        Some(match enode {
            // TrigLanguage::Pi => (NotNan::new(std::f64::consts::PI).unwrap(), "pi".parse().unwrap()),
            TrigLanguage::Constant(c) => (*c, format!("{}", c).parse().unwrap()),
            TrigLanguage::Add([a, b]) if can_multiply(x(a)?, x(b)?) => (
                x(a)? + x(b)?,
                format!("(+ {} {})", x(a)?, x(b)?).parse().unwrap(),
            ),
            TrigLanguage::Sub([a, b]) if can_multiply(x(a)?, x(b)?) => (
                x(a)? - x(b)?,
                format!("(- {} {})", x(a)?, x(b)?).parse().unwrap(),
            ),
            TrigLanguage::Mul([a, b]) if can_multiply(x(a)?, x(b)?) => (
                x(a)? * x(b)?,
                format!("(* {} {})", x(a)?, x(b)?).parse().unwrap(),
            ),
            TrigLanguage::Div([a, b]) if can_divide(x(a)?, x(b)?) => (
                x(a)? / x(b)?,
                format!("(/ {} {})", x(a)?, x(b)?).parse().unwrap(),
            ),
            TrigLanguage::Neg([a]) => (-x(a)?, format!("(~ {})", x(a)?).parse().unwrap()),
            // TrigLanguage::Pow([a, b]) => (
            //     NotNan::new(x(a)?.powf(x(b)?.into_inner())).unwrap(), // TODO: handle invalids
            //     format!("(pow {} {})", x(a)?, x(b)?).parse().unwrap(),
            // ),
            // TrigLanguage::Sqrt(a) => (
            //     NotNan::new(x(a)?.sqrt()).unwrap(), // TODO: handle invalids
            //     format!("(sqrt {})", x(a)?).parse().unwrap(),
            // ),
            // TrigLanguage::Cbrt(a) => (
            //     NotNan::new(x(a)?.cbrt()).unwrap(), // TODO: handle invalids
            //     format!("(cbrt {})", x(a)?).parse().unwrap(),
            // ),
            // TrigLanguage::Sin(a) => (
            //     NotNan::new(x(a)?.sin()).unwrap(), // TODO: handle invalids
            //     format!("(sin {})", x(a)?).parse().unwrap(),
            // ),
            // TrigLanguage::Cos(a) => (
            //     NotNan::new(x(a)?.cos()).unwrap(), // TODO: handle invalids
            //     format!("(cos {})", x(a)?).parse().unwrap(),
            // ),
            _ => return None,
        })
    }

    // fn pre_union(
    //         egraph: &egg::EGraph<TrigLanguage, Self>,
    //         id1: Id,
    //         id2: Id,
    //         justification: &Option<Justification>,
    //     ) {
    //     println!("pre_union: {:?} {:?}", id1, id2);
    //     println!("justification: {:?}", justification);
    // }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        merge_option(to, from, |a, b| {
            assert_eq!(a.0, b.0, "Merged non-equal constants");
            DidMerge(false, false)
        })
    }

    fn modify(egraph: &mut EGraph, id: Id) {
        let data = egraph[id].data.clone();
        if let Some((c, pat)) = data {
            if egraph.are_explanations_enabled() {
                egraph.union_instantiations(
                    &pat,
                    &format!("{}", c).parse().unwrap(),
                    &Default::default(),
                    "constant_fold".to_string(),
                );
            } else {
                let added = egraph.add(TrigLanguage::Constant(c));
                egraph.union(id, added);
            }
            // to not prune, comment this out
            egraph[id].nodes.retain(|n| n.is_leaf());

            #[cfg(debug_assertions)]
            egraph[id].assert_unique_leaves();
        }
    }
}

fn make_rules() -> Vec<Rewrite<TrigLanguage, ConstantFold>> {
    vec![
        // Commutativity
        rewrite!("+-commutative"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rewrite!("*-commutative"; "(* ?a ?b)" => "(* ?b ?a)"),
        // Associativity
        rewrite!("associate-+r+"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
        rewrite!("associate-+l+"; "(+ (+ ?a ?b) ?c)" => "(+ ?a (+ ?b ?c))"),
        rewrite!("associate-+r-"; "(+ ?a (- ?b ?c))" => "(- (+ ?a ?b) ?c)"),
        rewrite!("associate-+l-"; "(+ (- ?a ?b) ?c)" => "(- ?a (- ?b ?c))"),
        rewrite!("associate--r+"; "(- ?a (+ ?b ?c))" => "(- (- ?a ?b) ?c)"),
        rewrite!("associate--l+"; "(- (+ ?a ?b) ?c)" => "(+ ?a (- ?b ?c))"),
        rewrite!("associate--l-"; "(- (- ?a ?b) ?c)" => "(- ?a (+ ?b ?c))"),
        rewrite!("associate--r-"; "(- ?a (- ?b ?c))" => "(+ (- ?a ?b) ?c)"),
        rewrite!("associate-*r*"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"),
        rewrite!("associate-*l*"; "(* (* ?a ?b) ?c)" => "(* ?a (* ?b ?c))"),
        rewrite!("associate-*r/"; "(* ?a (/ ?b ?c))" => "(/ (* ?a ?b) ?c)"),
        rewrite!("associate-*l/"; "(* (/ ?a ?b) ?c)" => "(/ (* ?a ?c) ?b)"),
        rewrite!("associate-/r*"; "(/ ?a (* ?b ?c))" => "(/ (/ ?a ?b) ?c)"),
        rewrite!("associate-/r/"; "(/ ?a (/ ?b ?c))" => "(* (/ ?a ?b) ?c)"),
        rewrite!("associate-/l/"; "(/ (/ ?b ?c) ?a)" => "(/ ?b (* ?a ?c))"),
        rewrite!("associate-/l*"; "(/ (* ?b ?c) ?a)" => "(* ?b (/ ?c ?a))"),
        // Counting
        rewrite!("count-2"; "(+ ?x ?x)" => "(* 2 ?x)"),
        // Distributivity
        rewrite!("distribute-lft-in"; "(* ?a (+ ?b ?c))" => "(+ (* ?a ?b) (* ?a ?c))"),
        rewrite!("distribute-rgt-in"; "(* ?a (+ ?b ?c))" => "(+ (* ?b ?a) (* ?c ?a))"),
        rewrite!("distribute-lft-out"; "(+ (* ?a ?b) (* ?a ?c))" => "(* ?a (+ ?b ?c))"),
        rewrite!("distribute-lft-out--"; "(- (* ?a ?b) (* ?a ?c))" => "(* ?a (- ?b ?c))"),
        rewrite!("distribute-rgt-out"; "(+ (* ?b ?a) (* ?c ?a))" => "(* ?a (+ ?b ?c))"),
        rewrite!("distribute-rgt-out--"; "(- (* ?b ?a) (* ?c ?a))" => "(* ?a (- ?b ?c))"),
        rewrite!("distribute-lft1-in"; "(+ (* ?b ?a) ?a)" => "(* (+ ?b 1) ?a)"),
        rewrite!("distribute-rgt1-in"; "(+ ?a (* ?c ?a))" => "(* (+ ?c 1) ?a)"),
        // Distributivity Fp Safe
        rewrite!("distribute-lft-neg-in"; "(~ (* ?a ?b))" => "(* (~ ?a) ?b)"),
        rewrite!("distribute-rgt-neg-in"; "(~ (* ?a ?b))" => "(* ?a (~ ?b))"),
        rewrite!("distribute-lft-neg-out"; "(* (~ ?a) ?b)" => "(~ (* ?a ?b))"),
        rewrite!("distribute-rgt-neg-out"; "(* ?a (~ ?b))" => "(~ (* ?a ?b))"),
        rewrite!("distribute-neg-in"; "(~ (+ ?a ?b))" => "(+ (~ ?a) (~ ?b))"),
        rewrite!("distribute-neg-out"; "(+ (~ ?a) (~ ?b))" => "(~ (+ ?a ?b))"),
        rewrite!("distribute-frac-neg"; "(/ (~ ?a) ?b)" => "(~ (/ ?a ?b))"),
        rewrite!("distribute-frac-neg2"; "(/ ?a (~ ?b))" => "(~ (/ ?a ?b))"),
        rewrite!("distribute-neg-frac"; "(~ (/ ?a ?b))" => "(/ (~ ?a) ?b)"),
        rewrite!("distribute-neg-frac2"; "(~ (/ ?a ?b))" => "(/ ?a (~ ?b))"),
        // Cancel Sign Fp Safe
        rewrite!("cancel-sign-sub"; "(- ?a (* (~ ?b) ?c))" => "(+ ?a (* ?b ?c))"),
        rewrite!("cancel-sign-sub-inv"; "(- ?a (* ?b ?c))" => "(+ ?a (* (~ ?b) ?c))"),
        // Difference Of Squares Canonicalize
        rewrite!("swap-sqr"; "(* (* ?a ?b) (* ?a ?b))" => "(* (* ?a ?a) (* ?b ?b))"),
        rewrite!("unswap-sqr"; "(* (* ?a ?a) (* ?b ?b))" => "(* (* ?a ?b) (* ?a ?b))"),
        rewrite!("difference-of-squares"; "(- (* ?a ?a) (* ?b ?b))" => "(* (+ ?a ?b) (- ?a ?b))"),
        rewrite!("difference-of-sqr-1"; "(- (* ?a ?a) 1)" => "(* (+ ?a 1) (- ?a 1))"),
        rewrite!("difference-of-sqr--1"; "(+ (* ?a ?a) -1)" => "(* (+ ?a 1) (- ?a 1))"),
        rewrite!("pow-sqr"; "(* (pow ?a ?b) (pow ?a ?b))" => "(pow ?a (* 2 ?b))"),
        // // Sqr Pow Expand
        // rewrite!("sqr-pow"; "(pow ?a ?b)" => "(* (pow ?a (/ ?b 2)) (pow ?a (/ ?b 2)))"),
        // TODO: add conditional that a > 0

        // // Difference Of Squares Flip
        // rewrite!("flip-+"; "(+ ?a ?b)" => "(/ (- (* ?a ?a) (* ?b ?b)) (- ?a ?b))"),
        // rewrite!("flip--"; "(- ?a ?b)" => "(/ (- (* ?a ?a) (* ?b ?b)) (+ ?a ?b))"),
        // // TODO: causes issues since dem must not be zero, not so easy to check

        // Id Reduce
        rewrite!("remove-double-div"; "(/ 1 (/ 1 ?a))" => "?a"),
        rewrite!("rgt-mult-inverse"; "(* ?a (/ 1 ?a))" => "1" if is_not_zero("?a")),
        rewrite!("lft-mult-inverse"; "(* (/ 1 ?a) ?a)" => "1" if is_not_zero("?a")),
        // TODO: are the checks necessary?

        // Id Reduce Fp Safe Nan
        rewrite!("+-inverses"; "(- ?a ?a)" => "0"),
        rewrite!("div0"; "(/ 0 ?a)" => "0" if is_not_zero("?a")),
        rewrite!("mul0-lft"; "(* 0 ?a)" => "0"),
        rewrite!("mul0-rgt"; "(* ?a 0)" => "0"),
        rewrite!("*-inverses"; "(/ ?a ?a)" => "1" if is_not_zero("?a")),
        // Id Reduce Fp Safe
        rewrite!("+-lft-identity"; "(+ 0 ?a)" => "?a"),
        rewrite!("+-rgt-identity"; "(+ ?a 0)" => "?a"),
        rewrite!("--rgt-identity"; "(- ?a 0)" => "?a"),
        rewrite!("sub0-neg"; "(- 0 ?a)" => "(~ ?a)"),
        rewrite!("remove-double-neg"; "(~ (~ ?a))" => "?a"),
        rewrite!("*-lft-identity"; "(* 1 ?a)" => "?a"),
        rewrite!("*-rgt-identity"; "(* ?a 1)" => "?a"),
        rewrite!("/-rgt-identity"; "(/ ?a 1)" => "?a"),
        rewrite!("mul-1-neg"; "(* -1 ?a)" => "(~ ?a)"),
        // Nan Transform Fp Safe
        rewrite!("sub-neg"; "(- ?a ?b)" => "(+ ?a (~ ?b))"),
        rewrite!("unsub-neg"; "(+ ?a (~ ?b))" => "(- ?a ?b)"),
        rewrite!("neg-sub0"; "(~ ?b)" => "(- 0 ?b)"),
        rewrite!("neg-mul-1"; "(~ ?a)" => "(* -1 ?a)"),
        // Id Transform Safe
        rewrite!("div-inv"; "(/ ?a ?b)" => "(* ?a (/ 1 ?b))"),
        rewrite!("un-div-inv"; "(* ?a (/ 1 ?b))" => "(/ ?a ?b)"),
        // // Id Transform Clear Num
        // rewrite!("clear-num"; "(/ ?a ?b)" => "(/ 1 (/ ?b ?a))" if is_not_zero("?a")),
        // // TODO: Causes issues with trig functions; probably a confounding issue, need further
        // // investigation

        // Id Transform Fp Safe
        rewrite!("*-un-lft-identity"; "?a" => "(* 1 ?a)"),
        // Difference Of Cubes
        rewrite!("sum-cubes"; "(+ (pow ?a 3) (pow ?b 3))" => "(* (+ (* ?a ?a) (- (* ?b ?b) (* ?a ?b))) (+ ?a ?b))"),
        rewrite!("difference-cubes"; "(- (pow ?a 3) (pow ?b 3))" => "(* (+ (* ?a ?a) (+ (* ?b ?b) (* ?a ?b))) (- ?a ?b))"),
        // rewrite!("flip3-+"; "(+ ?a ?b)" => "(/ (+ (pow ?a 3) (pow ?b 3)) (+ (* ?a ?a) (- (* ?b ?b) (* ?a ?b))))"),
        // rewrite!("flip3--"; "(- ?a ?b)" => "(/ (- (pow ?a 3) (pow ?b 3)) (+ (* ?a ?a) (+ (* ?b ?b) (* ?a ?b))))"),

        // Fractions Distribute
        rewrite!("div-sub"; "(/ (- ?a ?b) ?c)" => "(- (/ ?a ?c) (/ ?b ?c))"),
        rewrite!("times-frac"; "(/ (* ?a ?b) (* ?c ?d))" => "(* (/ ?a ?c) (/ ?b ?d))"),
        // Fractions Transform
        rewrite!("sub-div"; "(- (/ ?a ?c) (/ ?b ?c))" => "(/ (- ?a ?b) ?c)"),
        rewrite!("frac-add"; "(+ (/ ?a ?b) (/ ?c ?d))" => "(/ (+ (* ?a ?d) (* ?b ?c)) (* ?b ?d))"),
        rewrite!("frac-sub"; "(- (/ ?a ?b) (/ ?c ?d))" => "(/ (- (* ?a ?d) (* ?b ?c)) (* ?b ?d))"),
        rewrite!("frac-times"; "(* (/ ?a ?b) (/ ?c ?d))" => "(/ (* ?a ?c) (* ?b ?d))"),
        rewrite!("frac-2neg"; "(/ ?a ?b)" => "(/ (~ ?a) (~ ?b))"),
        // Squares Reduce
        rewrite!("rem-square-sqrt"; "(* (sqrt ?x) (sqrt ?x))" => "?x"),
        // Squares Reduce Fp Sound
        rewrite!("sqr-neg"; "(* (~ ?x) (~ ?x))" => "(* ?x ?x)"),
        // Squares Transform Sound
        rewrite!("sqrt-pow2"; "(pow (sqrt ?x) ?y)" => "(pow ?x (/ ?y 2))"),
        rewrite!("sqrt-unprod"; "(* (sqrt ?x) (sqrt ?y))" => "(sqrt (* ?x ?y))"),
        rewrite!("sqrt-undiv"; "(/ (sqrt ?x) (sqrt ?y))" => "(sqrt (/ ?x ?y))"),
        // Sqrt Canonicalize
        rewrite!("sqrt-1"; "(sqrt 1)" => "1"),
        rewrite!("sqrt-0"; "(sqrt 0)" => "0"),
        rewrite!("sqrt-can"; "(/ (sqrt ?x) ?x)" => "(/ 1 (sqrt ?x))"),
        rewrite!("sqrt-can-inv"; "(/ ?x (sqrt ?x))" => "(sqrt ?x)"),
        rewrite!("sqrt-can-rev"; "(/ 1 (sqrt ?x))" => "(/ (sqrt ?x) ?x)"),
        // // Squares Transform
        // rewrite!("sqrt-pow1"; "(sqrt (pow ?x ?y))" => "(pow ?x (/ ?y 2))"),
        // rewrite!("sqrt-prod"; "(sqrt (* ?x ?y))" => "(* (sqrt ?x) (sqrt ?y))"),
        // rewrite!("sqrt-div"; "(sqrt (/ ?x ?y))" => "(/ (sqrt ?x) (sqrt ?y))"),
        rewrite!("add-sqr-sqrt"; "?x" => "(* (sqrt ?x) (sqrt ?x))" if is_non_negative_conservative("?x")),
        // TODO: determine if necessary, if so, then determine conditionals
        // conditions x and y are negative, which causes issues splitting them over real

        // Cubes Distribute
        rewrite!("cube-prod"; "(pow (* ?x ?y) 3)" => "(* (pow ?x 3) (pow ?y 3))"),
        rewrite!("cube-div"; "(pow (/ ?x ?y) 3)" => "(/ (pow ?x 3) (pow ?y 3))"),
        rewrite!("cube-mult"; "(pow ?x 3)" => "(* ?x (* ?x ?x))"),
        // Cubes Canonicalize
        rewrite!("cube-unmult"; "(* ?x (* ?x ?x))" => "(pow ?x 3)"),
        // Pow Reduce
        rewrite!("unpow-1"; "(pow ?a -1)" => "(/ 1 ?a)"),
        // Pow Reduce Fp Safe
        rewrite!("unpow1"; "(pow ?a 1)" => "?a"),
        // Pow Reduce Fp Safe Nan
        rewrite!("unpow0"; "(pow ?a 0)" => "1" if is_not_zero("?a")),
        rewrite!("pow-base-1"; "(pow 1 ?a)" => "1"),
        // Pow Expand Fp Safe
        rewrite!("pow1"; "?a" => "(pow ?a 1)"),
        // Pow Canonicalize
        rewrite!("unpow1/2"; "(pow ?a 0.5)" => "(sqrt ?a)"),
        rewrite!("unpow2"; "(pow ?a 2)" => "(* ?a ?a)"),
        rewrite!("unpow3"; "(pow ?a 3)" => "(* (* ?a ?a) ?a)"),
        rewrite!("pow-plus"; "(* (pow ?a ?b) ?a)" => "(pow ?a (+ ?b 1))"),
        // Pow Transform Sound
        rewrite!("pow-prod-down"; "(* (pow ?b ?a) (pow ?c ?a))" => "(pow (* ?b ?c) ?a)"),
        rewrite!("pow-prod-up"; "(* (pow ?a ?b) (pow ?a ?c))" => "(pow ?a (+ ?b ?c))"),
        rewrite!("pow-flip"; "(/ 1 (pow ?a ?b))" => "(pow ?a (~ ?b))"),
        rewrite!("pow-neg"; "(pow ?a (~ ?b))" => "(/ 1 (pow ?a ?b))" if is_not_zero("?a")),
        rewrite!("pow-div"; "(/ (pow ?a ?b) (pow ?a ?c))" => "(pow ?a (- ?b ?c))"),
        // Pow Specialize Sound
        rewrite!("pow1/2"; "(sqrt ?a)" => "(pow ?a 0.5)"),
        rewrite!("pow2"; "(* ?a ?a)" => "(pow ?a 2)"),
        rewrite!("pow3"; "(* (* ?a ?a) ?a)" => "(pow ?a 3)"),
        // // Pow Transform
        // rewrite!("pow-sub"; "(pow ?a (- ?b ?c))" => "(/ (pow ?a ?b) (pow ?a ?c))"),
        // rewrite!("pow-pow"; "(pow (pow ?a ?b) ?c)" => "(pow ?a (* ?b ?c))"),
        // rewrite!("pow-unpow"; "(pow ?a (* ?b ?c))" => "(pow (pow ?a ?b) ?c)"),
        // rewrite!("unpow-prod-up"; "(pow ?a (+ ?b ?c))" => "(* (pow ?a ?b) (pow ?a ?c))"),
        // rewrite!("unpow-prod-down"; "(pow (* ?b ?c) ?a)" => "(* (pow ?b ?a) (pow ?c ?a))"),
        // TODO: determine if necessary, if so, then determine conditionals

        // Pow Transform Fp Safe Nan
        rewrite!("pow-base-0"; "(pow 0 ?a)" => "0" if is_not_zero("?a")),
        // Pow Transform Fp Safe
        rewrite!("inv-pow"; "(/ 1 ?a)" => "(pow ?a -1)"),
        // Trig Reduce Fp Sound
        rewrite!("sin-0"; "(sin 0)" => "0"),
        rewrite!("cos-0"; "(cos 0)" => "1"),
        // Trig Reduce Fp Sound Nan
        rewrite!("sin-neg"; "(sin (~ ?x))" => "(~ (sin ?x))"),
        rewrite!("neg-sig"; "(~ (sin ?x))" => "(sin (~ ?x))"),
        rewrite!("cos-neg"; "(cos (~ ?x))" => "(cos ?x)"),
        rewrite!("neg-cos"; "(cos ?x)" => "(cos (~ ?x))"),
        // Trig Expand Fp Safe
        rewrite!("sqr-sin-b"; "(* (sin ?x) (sin ?x))" => "(- 1 (* (cos ?x) (cos ?x)))"),
        rewrite!("sqr-cos-b"; "(* (cos ?x) (cos ?x))" => "(- 1 (* (sin ?x) (sin ?x)))"),
        // Trig Reduce Sound
        rewrite!("cos-sin-sum"; "(+ (* (cos ?a) (cos ?a)) (* (sin ?a) (sin ?a)))" => "1"),
        rewrite!("1-sub-cos"; "(- 1 (* (cos ?a) (cos ?a)))" => "(* (sin ?a) (sin ?a))"),
        rewrite!("1-sub-sin"; "(- 1 (* (sin ?a) (sin ?a)))" => "(* (cos ?a) (cos ?a))"),
        rewrite!("-1-add-cos"; "(+ (* (cos ?a) (cos ?a)) -1)" => "(~ (* (sin ?a) (sin ?a)))"),
        rewrite!("-1-add-sin"; "(+ (* (sin ?a) (sin ?a)) -1)" => "(~ (* (cos ?a) (cos ?a)))"),
        rewrite!("sub-1-cos"; "(- (* (cos ?a) (cos ?a)) 1)" => "(~ (* (sin ?a) (sin ?a)))"),
        rewrite!("sub-1-sin"; "(- (* (sin ?a) (sin ?a)) 1)" => "(~ (* (cos ?a) (cos ?a)))"),
        rewrite!("sin-PI/6"; "(sin (/ (pi) 6))" => "0.5"),
        rewrite!("sin-PI/4"; "(sin (/ (pi) 4))" => "(/ (sqrt 2) 2)"),
        rewrite!("sin-PI*0.25"; "(sin (* (pi) 0.25))" => "(/ (sqrt 2) 2)"),
        rewrite!("sin-PI*-0.25"; "(sin (* (pi) -0.25))" => "(~ (/ (sqrt 2) 2))"),
        rewrite!("sin-PI/3"; "(sin (/ (pi) 3))" => "(/ (sqrt 3) 2)"),
        rewrite!("sin-PI/2"; "(sin (/ (pi) 2))" => "1"),
        rewrite!("sin-PI*0.5"; "(sin (* (pi) 0.5))" => "1"),
        rewrite!("sin-PI"; "(sin (pi))" => "0"),
        rewrite!("sin-+PI"; "(sin (+ ?x (pi)))" => "(~ (sin ?x))"),
        rewrite!("sin-+PI/2"; "(sin (+ ?x (/ (pi) 2)))" => "(cos ?x)"),
        rewrite!("cos-PI/6"; "(cos (/ (pi) 6))" => "(/ (sqrt 3) 2)"),
        rewrite!("cos-PI/4"; "(cos (/ (pi) 4))" => "(/ (sqrt 2) 2)"),
        rewrite!("cos-PI*0.25"; "(cos (* (pi) 0.25))" => "(/ (sqrt 2) 2)"),
        rewrite!("cos-PI/3"; "(cos (/ (pi) 3))" => "0.5"),
        rewrite!("cos-PI/2"; "(cos (/ (pi) 2))" => "0"),
        rewrite!("cos-PI*0.5"; "(cos (* (pi) 0.5))" => "0"),
        rewrite!("cos-PI"; "(cos (pi))" => "-1"),
        rewrite!("cos-+PI"; "(cos (+ ?x (pi)))" => "(~ (cos ?x))"),
        rewrite!("cos-+PI/2"; "(cos (+ ?x (* (pi) 0.5)))" => "(~ (sin ?x))"),
        rewrite!("hang-0p-tan"; "(/ (sin ?a) (+ 1 (cos ?a)))" => "(/ (sin (/ ?a 2)) (cos (/ ?a 2)))"),
        rewrite!("hang-0m-tan"; "(/ (~ (sin ?a)) (+ 1 (cos ?a)))" => "(/ (sin (/ (~ ?a) 2)) (cos (/ (~ ?a) 2)))"),
        rewrite!("hang-p0-tan"; "(/ (- 1 (cos ?a)) (sin ?a))" => "(/ (sin (/ ?a 2)) (cos (/ ?a 2)))"),
        rewrite!("hang-m0-tan"; "(/ (- 1 (cos ?a)) (~ (sin ?a)))" => "(/ (sin (/ (~ ?a) 2)) (cos (/ (~ ?a) 2)))"),
        rewrite!("tan-hang-0p"; "(/ (sin (* ?a 0.5)) (cos (* ?a 0.5)))" => "(/ (sin ?a) (+ 1 (cos ?a)))"),
        rewrite!("tan-hang-0m"; "(/ (sin (* (~ ?a) 0.5)) (cos (* (~ ?a) 0.5)))" => "(/ (~ (sin ?a)) (+ 1 (cos ?a)))"),
        rewrite!("tan-hang-p0"; "(/ (sin (* ?a 0.5)) (cos (* ?a 0.5)))" => "(/ (- 1 (cos ?a)) (sin ?a))"),
        rewrite!("tan-hang-m0"; "(/ (sin (* (~ ?a) 0.5)) (cos (* (~ ?a) 0.5)))" => "(/ (- 1 (cos ?a)) (~ (sin ?a)))" if is_not_zero("?a")),
        // Trig Expand Sound
        rewrite!("csc-cot"; "(/ 1 (* (sin ?a) (sin ?a)))" => "(+ 1 (/ (* (cos ?a) (cos ?a)) (* (sin ?a) (sin ?a))))"),
        rewrite!("sec-tan"; "(/ 1 (* (cos ?a) (cos ?a)))" => "(+ 1 (/ (* (sin ?a) (sin ?a)) (* (cos ?a) (cos ?a))))"),
        rewrite!("csc-sec"; "(* (/ 1 (* (cos ?a) (cos ?a))) (/ 1 (* (sin ?a) (sin ?a))))" => "(+ (/ 1 (* (cos ?a) (cos ?a))) (/ 1 (* (sin ?a) (sin ?a))))"),
        rewrite!("sin-sum"; "(sin (+ ?x ?y))" => "(+ (* (sin ?x) (cos ?y)) (* (cos ?x) (sin ?y)))"),
        rewrite!("cos-sum"; "(cos (+ ?x ?y))" => "(- (* (cos ?x) (cos ?y)) (* (sin ?x) (sin ?y)))"),
        // rewrite!("tan-sum"; "(/ (sin (+ ?a ?b)) (cos (+ ?a ?b)))" => "(/ (+ (/ (sin ?a) (cos ?a)) (/ (sin ?b) (cos ?b))) (- 1 (* (/ (sin ?a) (cos ?a)) (/ (sin ?b) (cos ?b)))))"),
        // rewrite!("cot-sum"; "(/ (cos (+ ?a ?b)) (sin (+ ?a ?b)))" => "(/ (- (* (/ (cos ?a) (sin ?a)) (/ (cos ?b) (sin ?b))) 1) (+ (/ (cos ?b) (sin ?b)) (/ (cos ?a) (sin ?a))))"),
        rewrite!("sin-diff"; "(sin (- ?x ?y))" => "(- (* (sin ?x) (cos ?y)) (* (cos ?x) (sin ?y)))"),
        rewrite!("cos-diff"; "(cos (- ?x ?y))" => "(+ (* (cos ?x) (cos ?y)) (* (sin ?x) (sin ?y)))"),
        rewrite!("sin-2"; "(sin (* 2 ?x))" => "(* 2 (* (sin ?x) (cos ?x)))"),
        rewrite!("sin-3"; "(sin (* 3 ?x))" => "(- (* 3 (sin ?x)) (* 4 (pow (sin ?x) 3)))"),
        rewrite!("2-sin"; "(* 2 (* (sin ?x) (cos ?x)))" => "(sin (* 2 ?x))"),
        rewrite!("3-sin"; "(- (* 3 (sin ?x)) (* 4 (pow (sin ?x) 3)))" => "(sin (* 3 ?x))"),
        rewrite!("cos-2"; "(cos (* 2 ?x))" => "(- (* (cos ?x) (cos ?x)) (* (sin ?x) (sin ?x)))"),
        rewrite!("cos-3"; "(cos (* 3 ?x))" => "(- (* 4 (pow (cos ?x) 3)) (* 3 (cos ?x)))"),
        rewrite!("2-cos"; "(- (* (cos ?x) (cos ?x)) (* (sin ?x) (sin ?x)))" => "(cos (* 2 ?x))"),
        rewrite!("3-cos"; "(- (* 4 (pow (cos ?x) 3)) (* 3 (cos ?x)))" => "(cos (* 3 ?x))"),
        // Trig Expand Sound2
        rewrite!("sqr-sin-a"; "(* (sin ?x) (sin ?x))" => "(- 0.5 (* 0.5 (cos (* 2 ?x))))"),
        rewrite!("sqr-cos-a"; "(* (cos ?x) (cos ?x))" => "(+ 0.5 (* 0.5 (cos (* 2 ?x))))"),
        rewrite!("diff-sin"; "(- (sin ?x) (sin ?y))" => "(* 2 (* (sin (/ (- ?x ?y) 2)) (cos (/ (+ ?x ?y) 2))))"),
        rewrite!("diff-cos"; "(- (cos ?x) (cos ?y))" => "(* -2 (* (sin (/ (- ?x ?y) 2)) (sin (/ (+ ?x ?y) 2))))"),
        rewrite!("sum-sin"; "(+ (sin ?x) (sin ?y))" => "(* 2 (* (sin (/ (+ ?x ?y) 2)) (cos (/ (- ?x ?y) 2))))"),
        rewrite!("sum-cos"; "(+ (cos ?x) (cos ?y))" => "(* 2 (* (cos (/ (+ ?x ?y) 2)) (cos (/ (- ?x ?y) 2))))"),
        rewrite!("cos-mult"; "(* (cos ?x) (cos ?y))" => "(/ (+ (cos (+ ?x ?y)) (cos (- ?x ?y))) 2)"),
        rewrite!("sin-mult"; "(* (sin ?x) (sin ?y))" => "(/ (- (cos (- ?x ?y)) (cos (+ ?x ?y))) 2)"),
        rewrite!("sin-cos-mult"; "(* (sin ?x) (cos ?y))" => "(/ (+ (sin (- ?x ?y)) (sin (+ ?x ?y))) 2)"),
        rewrite!("tan-2"; "(/ (sin (* 2 ?x)) (cos (* 2 ?x)))" => "(/ (* 2 (/ (sin ?x) (cos ?x))) (- 1 (* (/ (sin ?x) (cos ?x)) (/ (sin ?x) (cos ?x)))))"),
        rewrite!("2-tan"; "(/ (* 2 (/ (sin ?x) (cos ?x))) (- 1 (* (/ (sin ?x) (cos ?x)) (/ (sin ?x) (cos ?x)))))" => "(/ (sin (* 2 ?x)) (cos (* 2 ?x)))"),
    ]
}

fn to_egg_expr(expr: &Expression) -> RecExpr<TrigLanguage> {
    expr.to_string().parse().unwrap()
}

use crate::qgl::lexer::Lexer;
use crate::qgl::lexer::Token;

// pi -> Ident("pi")
// 5 -> Number(5)
// (pi) -> LParen, Ident("pi"), RParen
// (+ 1 2) -> LParen, Ident("+"), Number(1), Number(2), RParen
// (sin (pi)) -> LParen, Ident("sin"), LParen, Ident("pi"), RParen, RParen
// (+ (- 2 (pi)) 3) -> LParen, Ident("+"), LParen, Ident("-"), Number(2), LParen, Ident("pi"), RParen, RParen, Number(3), RParen
// [LParen, Ident("cos"), LParen, Op('*'), Number("0.5"), Ident("f1"), RParen, RParen]
fn _from_egg_expr(tokens: Vec<Token>) -> Expression {
    let (start, op_token) = if tokens[0] == Token::LParen {
        assert!(tokens.last() == Some(&Token::RParen));
        (1, tokens[1].clone())
    } else {
        (0, tokens[0].clone())
    };

    if let Token::Number(num) = op_token {
        return Expression::from_float(num.parse::<f64>().unwrap());
    }

    if let Token::Ident(ref id) = op_token {
        if id == "pi" {
            return Expression::Pi;
        }
        if id != "sin" && id != "cos" && id != "sqrt" && id != "pow" {
            return Expression::Variable(id.to_string());
        }
    }

    let mut operands = vec![];
    let mut i = start + 1;
    while i < tokens.len() {
        let token = &tokens[i];
        if *token == Token::LParen {
            let mut num_open_parenthesis = 1;
            let start = i + 1;
            for (j, token) in tokens.iter().enumerate().skip(i + 1) {
                if *token == Token::LParen {
                    num_open_parenthesis += 1;
                } else if *token == Token::RParen {
                    num_open_parenthesis -= 1;
                }

                if num_open_parenthesis == 0 {
                    operands.push(_from_egg_expr(tokens[start..j + 1].to_vec()));
                    i = j;
                    break;
                }
            }
        } else if *token == Token::RParen {
            assert_eq!(i, tokens.len() - 1);
        } else {
            operands.push(_from_egg_expr(tokens[i..i + 1].to_vec()));
        }
        i += 1;
    }

    match op_token {
        Token::Ident(id) => match id.clone().as_str() {
            "sin" => Expression::Sin(Box::new(operands[0].clone())),
            "cos" => Expression::Cos(Box::new(operands[0].clone())),
            "sqrt" => Expression::Sqrt(Box::new(operands[0].clone())),
            "pow" => Expression::Pow(Box::new(operands[0].clone()), Box::new(operands[1].clone())),
            _ => panic!("Invalid operator during parsing of egg expression"),
        },
        Token::Negation => {
            return Expression::Neg(Box::new(operands[0].clone()));
        }
        Token::Op(op) => match op {
            '+' => Expression::Add(Box::new(operands[0].clone()), Box::new(operands[1].clone())),
            '-' => Expression::Sub(Box::new(operands[0].clone()), Box::new(operands[1].clone())),
            '*' => Expression::Mul(Box::new(operands[0].clone()), Box::new(operands[1].clone())),
            '/' => Expression::Div(Box::new(operands[0].clone()), Box::new(operands[1].clone())),
            _ => panic!("Invalid operator during parsing of egg expression"),
        },
        _ => panic!("Invalid token during parsing of egg expression"),
    }
}

fn from_egg_expr(expr: RecExpr<TrigLanguage>) -> Expression {
    let expr_str = expr.to_string();
    let expr_tokens = Lexer::new(&expr_str).collect::<Vec<_>>();
    if expr_tokens.len() == 0 {
        panic!("Failure to lex expression: {}", expr_str);
    }

    // any Op('-') that is next to a Number is a negation and should be grouped
    let mut grouped_tokens = vec![];
    let mut i = 0;
    while i < expr_tokens.len() {
        if expr_tokens[i] == Token::Op('-') && i < expr_tokens.len() - 1 {
            if let Token::Number(n) = &expr_tokens[i + 1] {
                let n = n.parse::<f64>().unwrap();
                grouped_tokens.push(Token::Number((-n).to_string()));
                i += 1
            } else {
                grouped_tokens.push(expr_tokens[i].clone());
            }
        } else {
            grouped_tokens.push(expr_tokens[i].clone());
        }
        i += 1;
    }

    _from_egg_expr(grouped_tokens)
}

/// parse an expression, simplify it using egg, and pretty print it back out
pub fn simplify(expr: &Expression) -> Expression {
    // parse the expression, the type annotation tells it which Language to use
    let expr: RecExpr<TrigLanguage> = to_egg_expr(&expr);

    // simplify the expression using a Runner, which creates an e-graph with
    // the given expression and runs the given rules over it
    let runner = Runner::default().with_expr(&expr).run(&make_rules());
    let mut extractor = TrigExprExtractor::new(&runner.egraph);

    // the Runner knows which e-class the expression given with `with_expr` is in
    let root = runner.roots[0];

    // use an Extractor to pick the best element of the root eclass
    let best = extractor.extract_best(root);
    from_egg_expr(best)
}

#[allow(dead_code)]
pub fn extract_best_sine(expr: Expression) -> Option<Expression> {
    let expr: RecExpr<TrigLanguage> = to_egg_expr(&expr);
    let runner = Runner::default()
        .with_expr(&expr)
        .with_iter_limit(1000)
        .with_node_limit(10000000)
        .run(&make_rules());
    let egraph = &runner.egraph;
    let extractor = SineExtractor::new(egraph);
    let root = runner.roots[0];
    let best = extractor.extract_sine(root);
    best.map(from_egg_expr)
}

#[allow(dead_code)]
pub fn simplify_complex(expr: ComplexExpression) -> ComplexExpression {
    let ComplexExpression { real, imag } = expr;

    let real_expr: RecExpr<TrigLanguage> = to_egg_expr(&real);
    let imag_expr: RecExpr<TrigLanguage> = to_egg_expr(&imag);

    let runner = Runner::default()
        .with_expr(&real_expr)
        .with_expr(&imag_expr)
        .run(&make_rules());
    let mut extractor = TrigExprExtractor::new(&runner.egraph);

    // the Runner knows which e-class the expression given with `with_expr` is in
    let real_root = runner.roots[0];
    let imag_root = runner.roots[1];

    // use an Extractor to pick the best element of the root eclass
    let best_real = extractor.extract_best(real_root);
    let real_simple = from_egg_expr(best_real);

    let best_imag = extractor.extract_best(imag_root);
    let imag_simple = from_egg_expr(best_imag);

    ComplexExpression {
        real: real_simple,
        imag: imag_simple,
    }
}

#[allow(dead_code)]
pub fn simplify_matrix_no_context(
    matrix_expression: &Vec<Vec<ComplexExpression>>,
) -> Vec<Vec<ComplexExpression>> {
    let mut runner: Runner<TrigLanguage, ConstantFold> = Runner::default();

    for row in matrix_expression {
        for expr in row {
            let ComplexExpression { real, imag } = expr;
            let real_expr: RecExpr<TrigLanguage> = to_egg_expr(&real);
            let imag_expr: RecExpr<TrigLanguage> = to_egg_expr(&imag);
            runner = runner.with_expr(&real_expr).with_expr(&imag_expr);
        }
    }

    runner = runner.run(&make_rules());
    let extractor = Extractor::new(&runner.egraph, TrigCostFn);

    let mut simplified_matrix = vec![vec![]; matrix_expression.len()];
    let nrows = matrix_expression.len();
    let ncols = matrix_expression[0].len();

    for i in 0..nrows {
        for j in 0..ncols {
            let real_root = runner.roots[2 * (i * ncols + j)];
            let imag_root = runner.roots[2 * (i * ncols + j) + 1];

            let (_, best_real) = extractor.find_best(real_root);
            let real_simple = from_egg_expr(best_real);
            let (_, best_imag) = extractor.find_best(imag_root);
            let imag_simple = from_egg_expr(best_imag);

            simplified_matrix[i].push(ComplexExpression {
                real: real_simple,
                imag: imag_simple,
            });
        }
    }

    simplified_matrix
}

#[allow(dead_code)]
pub fn simplify_matrix(
    matrix_expression: &Vec<Vec<ComplexExpression>>,
) -> Vec<Vec<ComplexExpression>> {
    let mut runner: Runner<TrigLanguage, ConstantFold> = Runner::default();

    for row in matrix_expression {
        for expr in row {
            let ComplexExpression { real, imag } = expr;
            let real_expr: RecExpr<TrigLanguage> = to_egg_expr(&real);
            let imag_expr: RecExpr<TrigLanguage> = to_egg_expr(&imag);
            runner = runner.with_expr(&real_expr).with_expr(&imag_expr);
        }
    }

    runner = runner.run(&make_rules());
    let mut extractor = TrigExprExtractor::new(&runner.egraph);

    let mut simplified_matrix = vec![vec![]; matrix_expression.len()];
    let nrows = matrix_expression.len();
    let ncols = matrix_expression[0].len();

    for i in 0..nrows {
        for j in 0..ncols {
            let real_root = runner.roots[2 * (i * ncols + j)];
            let imag_root = runner.roots[2 * (i * ncols + j) + 1];

            let best_real = extractor.extract_best(real_root);
            let real_simple = from_egg_expr(best_real);
            let best_imag = extractor.extract_best(imag_root);
            let imag_simple = from_egg_expr(best_imag);

            simplified_matrix[i].push(ComplexExpression {
                real: real_simple,
                imag: imag_simple,
            });
        }
    }

    simplified_matrix
}

pub fn simplify_expressions_iter<'a>(
    expression: impl Iterator<Item = &'a Expression>,
) -> Vec<Expression> {
    let mut runner: Runner<TrigLanguage, ConstantFold> = Runner::default();

    let mut num_expressions = 0;
    for expr in expression {
        let expr: RecExpr<TrigLanguage> = to_egg_expr(&expr);
        runner = runner.with_expr(&expr);
        num_expressions += 1;
    }

    runner = runner.run(&make_rules());
    let mut extractor = TrigExprExtractor::new(&runner.egraph);

    let mut simplified_expressions = vec![];

    for i in 0..num_expressions {
        let root = runner.roots[i];
        let best = extractor.extract_best(root);
        simplified_expressions.push(from_egg_expr(best));
    }

    simplified_expressions
}

#[allow(dead_code)]
pub fn simplify_expressions(expression: Vec<Expression>) -> Vec<Expression> {
    let mut runner: Runner<TrigLanguage, ConstantFold> = Runner::default();

    let mut num_expressions = 0;
    for expr in expression {
        let expr: RecExpr<TrigLanguage> = to_egg_expr(&expr);
        runner = runner.with_expr(&expr);
        num_expressions += 1;
    }

    runner = runner.run(&make_rules());
    let mut extractor = TrigExprExtractor::new(&runner.egraph);

    let mut simplified_expressions = vec![];

    for i in 0..num_expressions {
        let root = runner.roots[i];
        let best = extractor.extract_best(root);
        simplified_expressions.push(from_egg_expr(best));
    }

    simplified_expressions
}

pub fn simplify_matrix_and_matvec(
    matrix_expression: &Vec<Vec<ComplexExpression>>,
    matvec_expression: &Vec<Vec<Vec<ComplexExpression>>>,
) -> (
    Vec<Vec<ComplexExpression>>,
    Vec<Vec<Vec<ComplexExpression>>>,
) {
    let mut runner: Runner<TrigLanguage, ConstantFold> = Runner::default();

    for row in matrix_expression {
        for expr in row {
            let ComplexExpression { real, imag } = expr;
            let real_expr: RecExpr<TrigLanguage> = to_egg_expr(real);
            let imag_expr: RecExpr<TrigLanguage> = to_egg_expr(imag);
            runner = runner.with_expr(&real_expr).with_expr(&imag_expr);
        }
    }

    for mat in matvec_expression {
        for row in mat {
            for expr in row {
                let ComplexExpression { real, imag } = expr;
                let real_expr: RecExpr<TrigLanguage> = to_egg_expr(real);
                let imag_expr: RecExpr<TrigLanguage> = to_egg_expr(imag);
                runner = runner.with_expr(&real_expr).with_expr(&imag_expr);
            }
        }
    }

    runner = runner.run(&make_rules());
    let mut extractor = TrigExprExtractor::new(&runner.egraph);

    let mut simplified_matrix = vec![vec![]; matrix_expression.len()];
    let nrows = matrix_expression.len();
    let ncols = matrix_expression[0].len();

    for i in 0..nrows {
        for j in 0..ncols {
            let real_root = runner.roots[2 * (i * ncols + j)];
            let imag_root = runner.roots[2 * (i * ncols + j) + 1];

            let best_real = extractor.extract_best(real_root);
            let real_simple = from_egg_expr(best_real);
            let best_imag = extractor.extract_best(imag_root);
            let imag_simple = from_egg_expr(best_imag);

            simplified_matrix[i].push(ComplexExpression {
                real: real_simple,
                imag: imag_simple,
            });
        }
    }

    let matrix_expr_offset = 2 * nrows * ncols;
    let nmats = matvec_expression.len();
    if nmats == 0 {
        return (simplified_matrix, vec![]);
    }
    let nrows = matvec_expression[0].len();
    let ncols = matvec_expression[0][0].len();
    let mut simplified_matvec = vec![vec![vec![]; nrows]; nmats];

    for m in 0..nmats {
        for i in 0..nrows {
            for j in 0..ncols {
                let real_root =
                    runner.roots[matrix_expr_offset + 2 * (m * nrows * ncols + i * ncols + j)];
                let imag_root =
                    runner.roots[matrix_expr_offset + 2 * (m * nrows * ncols + i * ncols + j) + 1];

                let best_real = extractor.extract_best(real_root);
                let real_simple = from_egg_expr(best_real);
                let best_imag = extractor.extract_best(imag_root);
                let imag_simple = from_egg_expr(best_imag);

                simplified_matvec[m][i].push(ComplexExpression {
                    real: real_simple,
                    imag: imag_simple,
                });
            }
        }
    }

    // println!("{:?}", context);

    (simplified_matrix, simplified_matvec)
}

// #[cfg(test)]
// use crate::UnitaryExpression;
// #[cfg(test)]
// use extract::BottomUpExtractor;

// #[test]
// fn test_simplify_matrix_and_matvec() {
//     let u3 = UnitaryExpression::new(
//         String::from("
//             utry U3(f1, f2, f3) {
//                 [
//                     [ cos(f1/2), ~e^(i*f3)*sin(f1/2) ],
//                     [ e^(i*f2)*sin(f1/2), e^(i*(f2+f3))*cos(f1/2) ]
//                 ]
//             }
//         "),
//     );
//     let grad = u3.differentiate();
//     let start = std::time::Instant::now();
//     let (simplified_matrix, simplified_matvec) = simplify_matrix_and_matvec(&u3.body, &grad.body);
//     let elapsed = start.elapsed();
//     println!("Time taken: {:?}", elapsed);
//     println!("{:?}", simplified_matrix);
//     println!("{:?}", simplified_matvec);

//     // bottom-up test
//     let matrix_expression = &u3.body;
//     let matvec_expression = &grad.body;

//     let mut runner: Runner<TrigLanguage, ConstantFold> = Runner::default();

//     for row in matrix_expression {
//         for expr in row {
//             let ComplexExpression { real, imag } = expr;
//             let real_expr: RecExpr<TrigLanguage> = to_egg_expr(real);
//             let imag_expr: RecExpr<TrigLanguage> = to_egg_expr(imag);
//             runner = runner.with_expr(&real_expr).with_expr(&imag_expr);
//         }
//     }

//     for mat in matvec_expression {
//         for row in mat {
//             for expr in row {
//                 let ComplexExpression { real, imag } = expr;
//                 let real_expr: RecExpr<TrigLanguage> = to_egg_expr(real);
//                 let imag_expr: RecExpr<TrigLanguage> = to_egg_expr(imag);
//                 runner = runner.with_expr(&real_expr).with_expr(&imag_expr);
//             }
//         }
//     }

//     runner = runner.run(&make_rules());
//     let extractor = BottomUpExtractor;
//     let start = std::time::Instant::now();
//     let _result = extractor.extract(&runner.egraph, &runner.roots);
//     let elapsed = start.elapsed();
//     println!("Time taken: {:?}", elapsed);
// }

#[allow(dead_code)]
pub fn check_many_equality(expr1s: &[&Expression], expr2s: &[&Expression]) -> bool {
    let expr1s: Vec<RecExpr<TrigLanguage>> = expr1s.iter().map(|expr| to_egg_expr(expr)).collect();
    let expr2s: Vec<RecExpr<TrigLanguage>> = expr2s.iter().map(|expr| to_egg_expr(expr)).collect();

    let mut runner: Runner<TrigLanguage, ConstantFold> = Runner::default()
        .with_iter_limit(120)
        .with_node_limit(100_000_000);
    for expr1 in &expr1s {
        runner = runner.with_expr(expr1);
    }
    for expr2 in &expr2s {
        runner = runner.with_expr(expr2);
    }
    runner = runner.run(&make_rules());

    for (expr1, expr2) in expr1s.iter().zip(expr2s.iter()) {
        if runner.egraph.equivs(expr1, expr2).len() == 0 {
            return false;
        }
    }

    true
}

#[allow(dead_code)]
pub fn check_equality(expr: &Expression, expr2: &Expression) -> bool {
    let expr1: RecExpr<TrigLanguage> = to_egg_expr(expr);
    let expr2: RecExpr<TrigLanguage> = to_egg_expr(expr2);

    let runner: Runner<TrigLanguage, ConstantFold> = Runner::default()
        .with_expr(&expr1)
        .with_expr(&expr2)
        .with_iter_limit(100)
        .with_node_limit(1_000_000)
        .run(&make_rules());

    runner.egraph.equivs(&expr1, &expr2).len() > 0
    // if runner.egraph.equivs(&expr1, &expr2).len() > 0 {
    //     return true;
    // }

    // let runner: Runner<TrigLanguage, ConstantFold> = Runner::default()
    //     .with_expr(&expr2)
    //     .with_iter_limit(40)
    //     .with_node_limit(25_000)
    //     .run(&make_rules());

    // runner.egraph.equivs(&expr2, &expr1).len() > 0
}

#[allow(dead_code)]
fn print_equality(s1: &str, s2: &str) {
    let expr1: RecExpr<TrigLanguage> = s1.parse().unwrap();
    let expr2: RecExpr<TrigLanguage> = s2.parse().unwrap();

    let mut runner: Runner<TrigLanguage, ConstantFold> = Runner::default()
        .with_explanations_enabled()
        .with_expr(&expr1)
        .run(&make_rules());
    println!(
        "{}",
        runner.explain_equivalence(&expr1, &expr2).get_flat_string()
    );
    // let egraph = runner.egraph;
    // let equivs = egraph.equivs(&expr1, &expr2);
    // if equivs.is_empty() {
    //     println!("{} and {} are not equivalent", s1, s2);
    // } else {
    //     println!("{} and {} are equivalent", s1, s2);
    // }
}

#[allow(dead_code)]
fn check_equality_lhs_only(s1: &str, s2: &str) -> bool {
    let expr1: RecExpr<TrigLanguage> = s1.parse().unwrap();
    let expr2: RecExpr<TrigLanguage> = s2.parse().unwrap();

    let runner: Runner<TrigLanguage, ConstantFold> =
        Runner::default().with_expr(&expr1).run(&make_rules());

    runner.egraph.equivs(&expr1, &expr2).len() > 0
}

#[allow(dead_code)]
fn check_equality_both(s1: &str, s2: &str) -> bool {
    let expr1: RecExpr<TrigLanguage> = s1.parse().unwrap();
    let expr2: RecExpr<TrigLanguage> = s2.parse().unwrap();

    let runner: Runner<TrigLanguage, ConstantFold> = Runner::default()
        .with_expr(&expr1)
        .with_node_limit(25_000)
        .run(&make_rules());

    let lhs = runner.egraph.equivs(&expr1, &expr2).len() > 0;

    let runner: Runner<TrigLanguage, ConstantFold> = Runner::default()
        .with_expr(&expr2)
        .with_node_limit(25_000)
        .run(&make_rules());

    let rhs = runner.egraph.equivs(&expr2, &expr1).len() > 0;

    lhs && rhs
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn check_equality_test() {
        // let s1 = "(* (sin (* (pi) (-0.25))) (cos (* (0.5) (- (?x2) (?x0)))))";
        // let s2 = "(* (pow (2) (-0.5)) (~ (cos (* (0.5) (- (?x0) (?x2))))))";
        // let s1 = "(* (sin (* pi (~ 0.25))) (cos (* (0.5) (- (?x2) (?x0)))))";
        // let s2 = "(* (pow 2 -0.5) (~ (cos (* (0.5) (- (?x0) (?x2))))))";
        // let s1 = "(/ (cos (* (0.5) (- (?x0) (?x2)))) (~ (sqrt (2))))";
        // let s2 = "(* (sin (* (pi) (-0.25))) (cos (* (0.5) (- (?x0) (?x2)))))";
        // let s1 = "(~ (/ (sin (x1)) (sqrt (2))))";
        // let s2 = "(/ (sin (x1)) (~ (sqrt (2))))";
        // let expr1: RecExpr<TrigLanguage> = s1.parse().unwrap();
        // let expr2: RecExpr<TrigLanguage> = s2.parse().unwrap();

        // let runner: Runner<TrigLanguage, ConstantFold> = Runner::default()
        //     .with_expr(&expr1)
        //     // .with_expr(&expr2)
        //     .with_iter_limit(50)
        //     .with_node_limit(50_000)
        //     .run(&make_rules());

        // println!("{:?}", runner.egraph.equivs(&expr1, &expr2).len());
        //
        // let s = "(* (sin (+ (x0) (x1))) (/ (- (cos (x0)) (sin (x0))) (sqrt (2))))";
        // let expr1: RecExpr<TrigLanguage> = s.parse().unwrap();
        // let mut runner: Runner<TrigLanguage, ConstantFold> = Runner::default()
        //     // .with_explanations_enabled()
        //     .with_expr(&expr1)
        //     .with_iter_limit(100)
        //     .with_node_limit(1_000_000)
        //     .run(&make_rules());

        // let expr1: RecExpr<TrigLanguage> = "0.001953125".parse().unwrap();
        // let expr2: RecExpr<TrigLanguage> = "7.450580596923828e-9".parse().unwrap();
        // println!("{}", runner.explain_equivalence(&expr1, &expr2).get_flat_string());
        // let runner = Runner::default().with_expr(&"(cos 0.5)".parse().unwrap()).run(&make_rules());
        // assert!(check_equality_lhs_only("(/ 1 4)", "(0.25)"));
        // assert!(check_equality_lhs_only("(/ (sqrt 2) 2)", "(/ 1 (sqrt 2))"));
        // assert!(check_equality_both("(sin (/ pi 4))", "(sin (* pi 0.25))"));
        // assert!(check_equality_lhs_only("(sin (/ pi 4))", "(/ 1 (sqrt 2))"));
        // assert!(check_equality_lhs_only("(sin (/ pi 4))", "(pow (2) (-0.5))"));
        // assert!(check_equality_both("(cos (* (0.5) (- (?x0) (?x2)))))", "(cos (* (0.5) (- (?x2) (?x0))))"));
        // assert!(check_equality_lhs_only("(* (sin (* (pi) (-0.25))) (cos (* (0.5) (- (?x2) (?x0)))))", "(* (pow (2) (-0.5)) (~ (cos (* (0.5) (- (?x0) (?x2))))))"));
        // (* (cos (x0)) (- (/ (cos (+ (x0) (x0))) (sqrt (2))) (/ (* (+ (cos (+ (x0) (x0))) (-1)) (sqrt (2))) (2))))
        // assert!(check_equality_both("(cos (* 2 ?x))", "(- (* (2) (* (cos ?x) (cos ?x))) 1)"));
        // assert!(check_equality_both("(- (cos (* 2 ?x)) 1)", "(* (-2) (* (sin ?x) (sin ?x)))"));
        // assert!(check_equality_both("(* (sin (~ ?x)) (sin (* 2 ?x)))", "(* (cos ?x) (- (cos (* 2 ?x)) 1))"));
        // assert!(check_equality_lhs_only("(* (cos ?x) (- (/ (cos (+ ?x ?x)) (sqrt 2)) (/ (* (+ (cos (+ ?x ?x)) (-1)) (sqrt (2))) (2))))", "(/ (cos ?x) (sqrt 2))"));

        // let s1 = "(* (sin (x0)) (- (/ (* (+ (1) (cos (+ (x0) (x0)))) (sqrt (2))) (2)) (/ (+ (* (-0.5) (sin (+ (x0) (x0)))) (- (+ (* (0.5) (sin (+ (x0) (x0)))) (1)) (-1))) (sqrt (2)))))";
        // let s2 = "(/ (sin (x0)) (sqrt (2)))";
        // let s1 = "(* (sin (x0)) (- (+ (1) (cos (+ (x0) (x0)))) (+ (* (-0.5) (sin (+ (x0) (x0)))) (- (+ (* (0.5) (sin (+ (x0) (x0)))) (1)) (-1))) ))";
        // let s2 = "(* (sin (x0)) (+ (cos (* 2 x0)) 1))";
        // let s1 = "(* (sin (x0)) (- (+ (1) (cos (+ (x0) (x0)))) (+ (* (-0.5) (sin (+ (x0) (x0)))) (- (+ (* (0.5) (sin (+ (x0) (x0)))) (1)) (-1)))))";
        // let s2 = "(* (sin (x0)) (- (cos (* 2 x0)) 1))";
        // assert!(check_equality_lhs_only(s1, s2));
        // let s1 = "(/ (+ (* (* (sin (* (pi) (0.25))) (cos (* (0.5) (- (θ) (λ))))) (/ (cos (θ)) (sqrt (2)))) (* (~ (* (sin (* (pi) (0.25))) (sin (* (0.5) (- (θ) (λ)))))) (/ (sin (θ)) (sqrt (2))))) (+ (* (/ (cos (θ)) (sqrt (2))) (/ (cos (θ)) (sqrt (2)))) (* (/ (sin (θ)) (sqrt (2))) (/ (sin (θ)) (sqrt (2))))))";
        // let s2 = "(/ (- (* (~ (* (sin (* (pi) (0.25))) (sin (* (0.5) (- (θ) (λ)))))) (/ (cos (θ)) (sqrt (2)))) (* (* (sin (* (pi) (0.25))) (cos (* (0.5) (- (θ) (λ))))) (/ (sin (θ)) (sqrt (2))))) (+ (* (/ (cos (θ)) (sqrt (2))) (/ (cos (θ)) (sqrt (2)))) (* (/ (sin (θ)) (sqrt (2))) (/ (sin (θ)) (sqrt (2))))))";
        // // let expr1: RecExpr<TrigLanguage> = s1.parse().unwrap();
        // let expr2: RecExpr<TrigLanguage> = s2.parse().unwrap();
        // let mut _runner: Runner<TrigLanguage, ConstantFold> = Runner::default()
        //     // .with_explanations_enabled()
        //     // .with_expr(&expr1)
        //     .with_expr(&expr2)
        //     .with_iter_limit(30)
        //     .with_node_limit(30_000)
        //     .run(&make_rules());
        // println!("{}", runner.explain_equivalence(&"1".parse().unwrap(), &"-1".parse().unwrap()).get_flat_string());
    }
}

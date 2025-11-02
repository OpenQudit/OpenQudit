use super::*;
use indexmap::IndexMap;
use rustc_hash::FxHashMap;

pub type Cost = NotNan<f64>;
pub const INFINITY: Cost = unsafe { NotNan::new_unchecked(std::f64::INFINITY) };

#[derive(Default, Debug, Clone)]
pub struct ExtractionResult {
    pub choices: IndexMap<Id, TrigLanguage>,
}

impl ExtractionResult {
    pub fn node_sum_cost(&self, _egraph: &EGraph, node: &TrigLanguage, costs: &FxHashMap<Id, Cost>) -> Cost {
        NotNan::new(node_cost(node)).unwrap()
            + node
                .children()
                .iter()
                .map(|child| {
                    costs.get(child).unwrap_or(&INFINITY)
                })
                .sum::<Cost>()
    }
    
    pub fn choose(&mut self, class_id: Id, node: TrigLanguage) {
        self.choices.insert(class_id, node);
    }
}

fn node_cost(enode: &TrigLanguage) -> f64 {
    let op_cost = match enode {
        TrigLanguage::Constant(_) => 0.5,
        TrigLanguage::Neg(_) => 1.0,
        TrigLanguage::Add(_) | TrigLanguage::Sub(_) => 1.0,
        TrigLanguage::Mul(_) | TrigLanguage::Div(_) => 5.0,
        TrigLanguage::Pow(_) | TrigLanguage::Sqrt(_) | TrigLanguage::Sin(_) | TrigLanguage::Cos(_) => 50.0,
        _ => 0.0,
    };
    op_cost
}

pub struct BottomUpExtractor;
impl BottomUpExtractor {
    pub fn extract(&self, egraph: &EGraph, _roots: &[Id]) -> ExtractionResult {
        let mut result = ExtractionResult::default();
        let mut costs = FxHashMap::<Id, Cost>::with_capacity_and_hasher(
            egraph.classes().len(),
            Default::default(),
        );
        let mut did_something = false;

        loop {
            for class in egraph.classes() {
                for node in &class.nodes {
                    let cost = result.node_sum_cost(egraph, node, &costs);
                    if &cost < costs.get(&class.id).unwrap_or(&INFINITY) {
                        result.choose(class.id.clone(), node.clone());
                        costs.insert(class.id.clone(), cost);
                        did_something = true;
                    }
                }
            }

            if did_something {
                did_something = false;
            } else {
                break;
            }
        }

        result
    }
}

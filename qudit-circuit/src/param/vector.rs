use std::collections::HashMap;

use qudit_core::ParamIndices;
use rustc_hash::FxHashMap;

use super::ArgumentList;
use super::Parameter;

pub type ParameterId = usize;
pub type ParameterIndex = usize;

/// Note: SlotMap is not used for persistent identifiers here as we want
/// extremely fast iteration over the parameters and simple flat ids for
/// the compact instruction data structure. To achieve this, id reusability
/// has to be sacrificed. This imposes a limit on the total number of
/// parameters that can ever be created with a single ParameterVector to
/// be `std::usize::MAX`.
#[derive(Clone, Debug, Default)]
pub struct ParameterVector {
    params: Vec<Parameter>,
    named_param_ids: HashMap<String, ParameterId>,
    id_to_index: FxHashMap<ParameterId, ParameterIndex>,
    ref_counts: FxHashMap<ParameterId, usize>,
    id_counter: ParameterId,
}

impl ParameterVector {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.params.len()
    }

    #[inline(always)]
    fn is_valid_id(&self, id: &ParameterId) -> bool {
        self.id_to_index.contains_key(id)
    }

    #[inline(always)]
    fn push(&mut self, param: Parameter) -> ParameterId {
        let id = {
            let new_id = self.id_counter;
            self.id_counter += 1;
            self.id_to_index.insert(new_id, self.params.len());
            self.ref_counts.insert(new_id, 1);
            new_id
        };

        if let Parameter::Named(name) = &param {
            self.named_param_ids.insert(name.clone(), id);
        }

        self.params.push(param);
        id
    }

    pub fn parse(&mut self, args: &ArgumentList) -> ParamIndices {
        if args.len() == 0 {
            return ParamIndices::Joint(0, 0);
        }

        let parameter_vector = args.parameters();

        let mut param_indices = vec![];
        let mut joint = true;
        for param in parameter_vector.into_iter() {
            if let Parameter::Named(name) = &param {
                // param_i is reserved variable name meant to reference unnamed variables
                if name.contains("param_") {
                    // Either extract an index i by matching to param_i
                    // or fail with param_ cannot be used in a parameter name
                    if let Some(s) = name.strip_prefix("param_") {
                        if let Ok(id) = s.parse::<ParameterId>() {
                            if self.is_valid_id(&id) {
                                param_indices.push(id);
                                joint = false;
                                continue
                            }
                        }
                    }

                    // TODO: Error handling
                    panic!("Cannot provide a parameter name containing 'param_'");
                }

                // Named variables that already exist do not need to be appended
                if let Some(id) = self.named_param_ids.get(name) {
                    param_indices.push(*id);
                    joint = false;
                    continue
                }
            }
            param_indices.push(self.push(param));
        }

        match joint {
            true => ParamIndices::Joint(param_indices[0], param_indices.len()),
            false => ParamIndices::Disjoint(param_indices),
        }
    } 
}

impl std::ops::Deref for ParameterVector {
    type Target = [Parameter];
    fn deref(&self) -> &Self::Target {
        &self.params
    }
}

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use qudit_core::{ComplexScalar, RealScalar};

use crate::expressions::{ExpressionBody, NamedExpression};
use crate::index::TensorIndex;
use crate::{DifferentiationLevel, GenerationShape, Module, ModuleBuilder, WriteFunc, FUNCTION, GRADIENT};
use crate::TensorExpression;
use qudit_core::c32;
use qudit_core::c64;

pub type ExpressionId = usize;

pub struct CachedExpressionBody {
    func: ExpressionBody,
    grad: Option<ExpressionBody>,
    hess: Option<ExpressionBody>,
}

impl CachedExpressionBody {
    pub fn num_elements(&self) -> usize {
        self.func.num_elements()
    }

    pub fn differentiate(&mut self, diff_level: DifferentiationLevel) {
        if diff_level == FUNCTION {
            return
        }

        if diff_level == GRADIENT {
            if self.grad.is_some() {
                return
            }

            

                // let mut grad_exprs = vec![];
                // for variable in &variables {
                //     for expr in &exprs {
                //         let grad_expr = expr.differentiate(&variable);
                //         grad_exprs.push(grad_expr);
                //     }
                // }

                // let mut hess_exprs = vec![];
                // for variable in &variables {
                //     for expr in &grad_exprs {
                //         let hess_expr = expr.differentiate(&variable);
                //         hess_exprs.push(hess_expr);
                //     }
                // }

                // let simplified_exprs = simplify_expressions(exprs
                //     .into_iter()
                //     .chain(grad_exprs.into_iter())
                //     .chain(hess_exprs.into_iter())
                //     .collect());
        }
    }
}

impl<B: Into<ExpressionBody>> From<B> for CachedExpressionBody {
    fn from(value: B) -> Self {
        CachedExpressionBody {
            func: value.into(),
            grad: None,
            hess: None,
        }
    }
}

pub struct CachedTensorExpression {
    name: String,
    variables: Vec<String>,
    indices: Vec<TensorIndex>,
    expressions: CachedExpressionBody,
    id_lookup: BTreeMap<ExpressionId, (Vec<usize>, GenerationShape)>
}

impl CachedTensorExpression {
    pub fn new<E: Into<TensorExpression>>(expr: E, id: ExpressionId) -> Self {
        let expr = expr.into();
        let mut id_lookup = BTreeMap::new();
        id_lookup.insert(id, ((0..expr.rank()).collect(), expr.indices().into()));
        let (name, variables, body, indices) = expr.destruct();
        CachedTensorExpression {
            name,
            variables,
            indices,
            expressions: body.into(),
            id_lookup,
        }
    }

    pub fn differentiate(&mut self, diff_level: DifferentiationLevel) {
        self.expressions.differentate(diff_level)
    }

    pub fn num_elements(&self) -> usize {
        self.expressions.num_elements()
    }

    pub fn form_expression(&self) -> TensorExpression {
        let named = NamedExpression::new(self.name.clone(), self.variables.clone(), self.expressions.func.clone());
        TensorExpression::from_raw(self.indices.clone(), named)
    }

    pub fn form_modified_indices(&self, modifiers: &(Vec<usize>, GenerationShape)) -> Vec<TensorIndex> {
        let perm_index_sizes: Vec<usize> = modifiers.0.iter().map(|p| self.indices[*p].index_size()).collect();
        let redirection = modifiers.1.calculate_directions(&perm_index_sizes);
        perm_index_sizes.iter().zip(redirection.iter()).enumerate().map(|(id, (s, d))| TensorIndex::new(*d, id, *s)).collect()
    }

    pub fn form_modified_expression(&self, modifiers: &(Vec<usize>, GenerationShape)) -> TensorExpression {
        let perm_index_sizes: Vec<usize> = modifiers.0.iter().map(|p| self.indices[*p].index_size()).collect();
        let redirection = modifiers.1.calculate_directions(&perm_index_sizes);
        let mut expression = self.form_expression();
        let new_indices = perm_index_sizes.iter().zip(redirection.iter()).enumerate().map(|(id, (s, d))| TensorIndex::new(*d, id, *s)).collect();
        expression.permute(&modifiers.0, redirection);
        expression.reindex(new_indices);
        expression
    }
}

#[derive(Default)]
pub struct ExpressionCache {
    expressions: BTreeMap<ExpressionId, CachedTensorExpression>,
    id_lookup: BTreeMap<ExpressionId, ExpressionId>, // Maps derived expressions to their base id
    name_lookup: BTreeMap<String, Vec<ExpressionId>>, // Name to base expression ids
    module32: Option<Module<f32>>,
    module64: Option<Module<f64>>,
    id_counter: ExpressionId
}

impl ExpressionCache {
    pub fn new() -> Self {
        Self {
            expressions: BTreeMap::new(),
            id_lookup: BTreeMap::new(),
            name_lookup: BTreeMap::new(),
            module32: None,
            module64: None,
            id_counter: 0,
        }
    }

    pub fn new_shared() -> Rc<RefCell<Self>> {
        Rc::new(Self::new().into())
    }

    pub fn lookup(&self, expr: impl AsRef<NamedExpression>) -> Option<ExpressionId> {
        let expr = expr.as_ref();
        if let Some(ids) = self.name_lookup.get(expr.name()) {
            for id in ids {
                let cexpr = self.expressions.get(id).expect("Expected since just looked up id.");
                if expr == &cexpr.expressions.func {
                    return Some(*id);
                }
            }
        }
        None
    }

    pub fn contains(&self, expr: impl AsRef<NamedExpression>) -> bool {
        self.lookup(expr).is_some()
    }

    fn get_new_id(&mut self) -> ExpressionId {
        let new_id = self.id_counter;
        self.id_counter += 1;
        new_id
    }

    fn uncompile(&mut self) {
        self.module32 = None;
        self.module64 = None;
    }

    pub fn insert(&mut self, expr: impl Into<TensorExpression>) -> ExpressionId {
        let expr = expr.into();
        if let Some(id) = self.lookup(&expr) {
            return id;
        }

        self.uncompile();

        let id = self.get_new_id();
        self.name_lookup.insert(expr.name().to_owned(), vec![id]);
        self.expressions.insert(id, CachedTensorExpression::new(expr, id));
        self.id_lookup.insert(id, id);
        id
    }

    pub fn indices(&self, expr_id: ExpressionId) -> Vec<TensorIndex> {
        let base_id = self.id_lookup.get(&expr_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));
        let cexpr = self.expressions.get(base_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));
        let modifiers = cexpr.id_lookup.get(&expr_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));
        cexpr.form_modified_indices(modifiers)
    }

    pub fn num_elements(&self, expr_id: ExpressionId) -> usize {
        let base_id = self.id_lookup.get(&expr_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));
        let cexpr = self.expressions.get(base_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));
        cexpr.num_elements()
    }

    pub fn generation_shape(&self, expr_id: ExpressionId) -> GenerationShape {
        let base_id = self.id_lookup.get(&expr_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));
        let cexpr = self.expressions.get(base_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));
        let modifiers = cexpr.id_lookup.get(&expr_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));
        modifiers.1
    }
    
    pub fn base_name(&self, expr_id: ExpressionId) -> String {
        let base_id = self.id_lookup.get(&expr_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));
        let cexpr = self.expressions.get(base_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));
        
       cexpr.name.clone()
    }

    pub fn name(&self, expr_id: ExpressionId) -> String {
        let base_id = self.id_lookup.get(&expr_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));
        let cexpr = self.expressions.get(base_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));
        let modifiers = cexpr.id_lookup.get(&expr_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));

        let base_name = cexpr.name.clone();
        let permutation = modifiers.0.iter().map(|&i| i.to_string()).collect::<Vec<String>>().join("_");
        let shape = modifiers.1.to_vec().iter().map(|&i| i.to_string()).collect::<Vec<String>>().join("_");
        format!("{base_name}_perm{permutation}_shape{shape}")
    }
    
    pub fn trace(&mut self, expr_id: ExpressionId, pairs: Vec<(usize, usize)>) -> ExpressionId {
        // expr_id -> base_id -> base_expr -> derived_name -> base_name -> check if base_name_tracedXYZ exists
        let old_name = self.name(expr_id);
        let traced = pairs.iter().map(|(x, y)| format!("{x}_{y}")).collect::<Vec<String>>().join("_");
        let traced_name = format!("traced{traced}_{old_name}");
        
        if let Some(ids) = self.name_lookup.get(&traced_name) {
            if ids.len() == 1 {
                return ids[0];
            }    
            // TODO: add some more checks here and handle this case properly
            unimplemented!("Very unlikely to reach here, if you do please file a report.");
        }

        self.uncompile();

        let base_id = self.id_lookup.get(&expr_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));
        let cexpr = self.expressions.get(base_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));
        let modifiers = cexpr.id_lookup.get(&expr_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));

        let old_expr: TensorExpression = cexpr.form_modified_expression(modifiers);
        let mut traced_expr = old_expr.partial_trace(&pairs);
        traced_expr.set_name(traced_name);
        
        self.insert(traced_expr)
    }

    pub fn permute_reshape(&mut self, expr_id: ExpressionId, perm: Vec<usize>, reshape: GenerationShape) -> ExpressionId {
        let base_id = *self.id_lookup.get(&expr_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));
        let cexpr = self.expressions.get(&base_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));
        let modifiers = cexpr.id_lookup.get(&expr_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));

        // assert reshape num elements is correct
        let composed_perm: Vec<usize> = perm.iter().map(|&idx| modifiers.0[idx]).collect(); // TODO: check if correct ordering

        let new_val = (composed_perm, reshape);
        for (id, val) in cexpr.id_lookup.iter() {
            if *val == new_val {
                return *id
            }
        }

        let new_id = self.get_new_id();
        let cexpr = self.expressions.get_mut(&base_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));
        cexpr.id_lookup.insert(new_id, new_val);
        self.id_lookup.insert(new_id, base_id);
        new_id
    }

    fn _get_module<R: RealScalar>(&self) -> &Option<Module<R>> {
        if std::any::TypeId::of::<R>() == std::any::TypeId::of::<f32>() {
            // Safety: Just checked exact type id
            unsafe { std::mem::transmute(&self.module32) }
        } else if std::any::TypeId::of::<R>() == std::any::TypeId::of::<f64>() {
            // Safety: Just checked exact type id
            unsafe { std::mem::transmute(&self.module64) }
        } else {
            unreachable!()
        }
    }

    pub fn is_compiled<R: RealScalar>(&self) -> bool {
        self._get_module::<R>().is_some()
    }

    pub fn differentiate(&mut self, diff_level: DifferentiationLevel) {
        for (_, cexpr) in self.expressions {
            cexpr.differentiate(diff_level)
        }
    }

    pub fn compile<R: RealScalar>(&mut self) {
        // create module builder
        let module_builder: ModuleBuilder<R> = ModuleBuilder::new("cache");
        // For each base expression
            // create compilation unit
            // add to module builder
        // build and place in the place I want
        todo!()
    }

    pub fn get_fn<R: RealScalar>(&mut self, expr_id: ExpressionId) -> WriteFunc<R> {
        if !self.is_compiled::<R>() {
            self.compile::<R>();
        }
        let module = self._get_module().as_ref().expect("Module should exist due to previous compilation.");
        // Safety: will compile for all expressions before getting this function
        // TODO: another safety value needs to expressed: WriteFunc is only valid while we haven't
        // compiled again
        unsafe {
            module.get_function_raw(&self.base_name(expr_id))
        }
    }

    pub fn get_output_map<R: RealScalar>(&self, expr_id: ExpressionId) -> Vec<u64> {
        todo!()
    }
}


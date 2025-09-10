use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use qudit_core::{ComplexScalar, RealScalar};

use crate::analysis::{simplify_expressions, simplify_expressions_iter};
use crate::codegen::CompilableUnit;
use crate::expressions::{ExpressionBody, NamedExpression};
use crate::index::{IndexSize, TensorIndex};
use crate::{ComplexExpression, DifferentiationLevel, Expression, GenerationShape, Module, ModuleBuilder, WriteFunc, FUNCTION, GRADIENT, HESSIAN};
use crate::TensorExpression;
use qudit_core::c32;
use qudit_core::c64;

pub type ExpressionId = usize;

pub struct CachedExpressionBody {
    original: ExpressionBody,
    func: Option<Vec<Expression>>,
    grad: Option<Vec<Expression>>,
    hess: Option<Vec<Expression>>,
}

impl CachedExpressionBody {
    pub fn new(original: impl Into<ExpressionBody>) -> CachedExpressionBody {
        let original = original.into();
        CachedExpressionBody {
            original,
            func: None,
            grad: None,
            hess: None,
        }
    }


    pub fn num_elements(&self) -> usize {
        self.original.num_elements()
    }

    // Simplify and differentiate to prepare expression to evaluate up to diff_level.
    pub fn prepare(&mut self, variables: &[String], diff_level: DifferentiationLevel) -> bool {
        let mut has_changed = false;

        if self.func.is_none() {
            self.func = Some(simplify_expressions_iter(
                self.original.iter()
                .flat_map(|c| [&c.real, &c.imag])
            ));

            has_changed = true;
        }

        if diff_level >= GRADIENT && self.grad.is_none() {
            let mut grad_exprs = vec![];
            for variable in variables {
                for expr in self.original.iter() {
                    let grad_expr = expr.differentiate(&variable);
                    grad_exprs.push(grad_expr);
                }
            }

            self.grad = Some(simplify_expressions_iter(
                self.original.iter()
                .chain(grad_exprs.iter())
                .flat_map(|c| [&c.real, &c.imag])
            ));

            has_changed = true;
        }

        if diff_level >= HESSIAN && self.hess.is_some() {
            let mut grad_exprs = vec![];
            for variable in variables {
                for expr in self.original.iter() {
                    let grad_expr = expr.differentiate(&variable);
                    grad_exprs.push(grad_expr);
                }
            }

            let mut hess_exprs = vec![];
            for variable in variables {
                for expr in grad_exprs.iter() {
                    let hess_expr = expr.differentiate(&variable);
                    hess_exprs.push(hess_expr);
                }
            }

            self.hess = Some(simplify_expressions_iter(
                self.original.iter()
                .chain(grad_exprs.iter())
                .chain(hess_exprs.iter())
                .flat_map(|c| [&c.real, &c.imag])
            ));

            has_changed = true;
        }

        has_changed
    }
}

impl<B: Into<ExpressionBody>> From<B> for CachedExpressionBody {
    fn from(value: B) -> Self {
        CachedExpressionBody::new(value)
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

    // TODO: deduplicate code with tensorexpression
    pub fn indices(&self) -> &[TensorIndex] {
        &self.indices
    }

    // TODO: deduplicate code with tensorexpression
    pub fn rank(&self) -> usize {
        self.indices.len()
    }

    pub fn num_params(&self) -> usize {
        self.variables.len()
    }

    // TODO: deduplicate code with tensorexpression
    pub fn dimensions(&self) -> Vec<IndexSize> {
        self.indices.iter().map(|idx| idx.index_size()).collect()
    }

    // TODO: deduplicate code with tensorexpression
    pub fn tensor_strides(&self) -> Vec<usize> {
        let mut strides = Vec::with_capacity(self.indices.len());
        let mut current_stride = 1;
        for &index in self.indices.iter().rev() {
            strides.push(current_stride);
            current_stride *= index.index_size();
        }
        strides.reverse();
        strides
    }

    pub fn elements(&self) -> &[ComplexExpression] {
        &self.expressions.original
    }


    // Simplify and differentiate to prepare expression to evaluate up to diff_level.
    pub fn prepare(&mut self, diff_level: DifferentiationLevel) -> bool {
        self.expressions.prepare(&self.variables, diff_level)
    }

    fn process_name_for_gen(&self, name: String) -> String {
        name.replace(" ", "_")
            .replace("⊗", "t")
            .replace("†", "d")
            .replace("^", "p")
            .replace("⋅", "x")
    }

    fn add_to_builder<'a, R: RealScalar>(&'a self, mut builder: ModuleBuilder<'a, R>) -> ModuleBuilder<'a, R> {

        if self.expressions.func.is_some() {
            // println!("Adding {} function to module", self.name.clone() + "_" + "1");
            let unit = CompilableUnit::new(
                &(self.process_name_for_gen(self.name.clone() + "_" + "1")),
                self.expressions.func.as_ref().unwrap(),
                self.variables.clone(),
                self.expressions.original.len() * 2,
            );

            builder = builder.add_unit(unit);
        }

        if self.expressions.grad.is_some() {
            // println!("Adding {} function to module", self.name.clone() + "_" + "2");
            let unit = CompilableUnit::new(
                &(self.process_name_for_gen(self.name.clone() + "_" + "2")),
                self.expressions.grad.as_ref().unwrap(),
                self.variables.clone(),
                self.expressions.original.len() * 2,
            );

            builder = builder.add_unit(unit);
        }

        if self.expressions.hess.is_some() {
            // println!("Adding {} function to module", self.name.clone() + "_" + "3");
            let unit = CompilableUnit::new(
                &(self.process_name_for_gen(self.name.clone() + "_" + "3")),
                self.expressions.grad.as_ref().unwrap(),
                self.variables.clone(),
                self.expressions.original.len() * 2,
            );

            builder = builder.add_unit(unit);
        }
        
        builder
    }

    pub fn num_elements(&self) -> usize {
        self.expressions.num_elements()
    }

    pub fn form_expression(&self) -> TensorExpression {
        let named = NamedExpression::new(self.name.clone(), self.variables.clone(), self.expressions.original.clone());
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
                if expr == &cexpr.expressions.original {
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
        let expr: TensorExpression = expr.into();
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

    pub fn num_params(&self, expr_id: ExpressionId) -> usize {
        let base_id = self.id_lookup.get(&expr_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));
        let cexpr = self.expressions.get(base_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));
        cexpr.num_params()
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

    fn _get_module_mut<R: RealScalar>(&mut self) -> &mut Option<Module<R>> {
        if std::any::TypeId::of::<R>() == std::any::TypeId::of::<f32>() {
            // Safety: Just checked exact type id
            unsafe { std::mem::transmute(&mut self.module32) }
        } else if std::any::TypeId::of::<R>() == std::any::TypeId::of::<f64>() {
            // Safety: Just checked exact type id
            unsafe { std::mem::transmute(&mut self.module64) }
        } else {
            unreachable!()
        }
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

    // Simplify and differentiate to prepare expression to evaluate up to diff_level.
    pub fn prepare(&mut self, diff_level: DifferentiationLevel) {
        let mut should_uncompile = false;
        for (_, cexpr) in self.expressions.iter_mut() {
            if cexpr.prepare(diff_level) {
                should_uncompile = true;
            }
        }

        if should_uncompile {
            self.uncompile();
        }
    }

    pub fn compile<R: RealScalar>(&mut self) {
        let mut module_builder: ModuleBuilder<R> = ModuleBuilder::new("cache");
        // For each base expression
        for (_, cexpr) in self.expressions.iter() {
            module_builder = cexpr.add_to_builder(module_builder);
        }
        *self._get_module_mut() = Some(module_builder.build());
    }

    // TODO: explore a lock feature: compile and lock -> returns something that needs to be passed
    // into get_fn; that provides guarantees that when the lock goes out of scope all the write
    // functions are also out of scope.

    pub fn get_fn<R: RealScalar>(&mut self, expr_id: ExpressionId, diff_level: DifferentiationLevel) -> WriteFunc<R> {
        if !self.is_compiled::<R>() {
            self.compile::<R>();
        }
        let module = self._get_module().as_ref().expect("Module should exist due to previous compilation.");
        // Safety: will compile for all expressions before getting this function
        unsafe {
            // TODO: ensure diff_level function exists
            module.get_function_raw(&(self.base_name(expr_id) + "_" + &diff_level.to_string()))
        }
    }

    pub fn get_output_map<R: RealScalar>(
        &self,
        expr_id: ExpressionId,
        row_stride: u64,
        col_stride: u64,
        mat_stride: u64,
    ) -> Vec<u64> {
        let base_id = self.id_lookup.get(&expr_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));
        let cexpr = self.expressions.get(base_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));
        let modifiers = cexpr.id_lookup.get(&expr_id)
            .unwrap_or_else(|| panic!("Failed to {expr_id} in cache."));

        let (perm, shape) = modifiers;

        // TODO: deduplicate code with tensorexpression.permute; extract calc_elem_perm ->
        // Vec<TensorIndex>
        let original_strides = cexpr.tensor_strides();
        let original_dimensions = cexpr.dimensions();
        
        let new_strides = {
            let mut strides = Vec::with_capacity(cexpr.rank());
            let mut current_stride = 1;
            for perm_index in perm.iter().map(|p| cexpr.indices()[*p]).rev() {
                strides.push(current_stride);
                current_stride *= perm_index.index_size();
            }
            strides.reverse();
            strides
        };

        let mut complex_elem_perm = Vec::with_capacity(cexpr.num_elements());
        for i in 0..cexpr.num_elements() {
            let mut original_coordinate: Vec<usize> = Vec::with_capacity(cexpr.rank());
            let mut temp_i = i;
            for d_idx in 0..cexpr.rank() {
                original_coordinate.push((temp_i / original_strides[d_idx]) % original_dimensions[d_idx]);
                temp_i %= original_strides[d_idx]; // Update temp_i for next dimension
            }

            // Map original coordinate components to their new positions according to `perm`.
            // If `perm[j]` is `k`, it means the `j`-th dimension in the new order
            // corresponds to the `k`-th dimension in the original order.
            let mut permuted_coordinate: Vec<usize> = vec![0; cexpr.rank()];
            for j in 0..cexpr.rank() {
                permuted_coordinate[j] = original_coordinate[perm[j]];
            }

            // Calculate new linear index using the permuted coordinate and new strides
            let mut new_linear_idx = 0;
            for d_idx in 0..cexpr.rank() {
                new_linear_idx += permuted_coordinate[d_idx] * new_strides[d_idx];
            }
            complex_elem_perm.push(new_linear_idx);
        }

        let ncols = shape.ncols() as u64;
        let nrows = shape.nrows() as u64;

        let num_real_elements = cexpr.num_elements() * 2;
        let mut map = Vec::with_capacity(num_real_elements);

        for real_idx in 0..(num_real_elements as u64) {
            let complex_idx = real_idx / 2;
            let complex_perm_idx = complex_elem_perm[complex_idx as usize] as u64;
            let imag_offset = real_idx % 2;
            let mat_idx = complex_perm_idx / (nrows * ncols);
            let row_idx = (complex_perm_idx % (nrows * ncols)) / ncols;
            let col_idx = complex_perm_idx % ncols;
            map.push(2 * (mat_idx * mat_stride + row_idx * row_stride + col_idx * col_stride) + imag_offset)
        }
        
        map
    }
}


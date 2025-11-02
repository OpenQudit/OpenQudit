#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u64)]
pub enum DirectiveOperation {
    Barrier = 0,
}

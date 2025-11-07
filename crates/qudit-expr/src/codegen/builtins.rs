use inkwell::intrinsics::Intrinsic;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Builtins {
    COS,
    SIN,
    SQRT,
    POW,
}

impl Builtins {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "cos" => Some(Self::COS),
            "sin" => Some(Self::SIN),
            "sqrt" => Some(Self::SQRT),
            "pow" => Some(Self::POW),
            "powi" => Some(Self::POW),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub fn to_str(self) -> &'static str {
        match self {
            Self::COS => "cos",
            Self::SIN => "sin",
            Self::SQRT => "sqrt",
            Self::POW => "pow",
        }
    }

    #[allow(dead_code)]
    pub fn arity(&self) -> usize {
        match self {
            Self::COS | Self::SIN | Self::SQRT => 1,
            Self::POW => 2,
        }
    }

    // pub fn check_args(&self, args: &Vec<TypedExpression>) {
    //     if args.len() != self.arity() {
    //         panic!("{} expects {} arguments, got {}", self.to_str(), self.arity(), args.len());
    //     }

    //     match self {
    //         Self::COS | Self::SIN | Self::TAN | Self::SQRT | Self::LOG | Self::EXP => {
    //             if args[0].dimension() != 0 {
    //                 panic!("{} expects scalar argument, got {}-dimensional matrix", self.to_str(), args[0].dimension());
    //             }
    //         },
    //         Self::POW => {
    //             if args[0].dimension() != 0 || args[1].dimension() != 0 {
    //                 panic!("{} expects scalar arguments, got {}-dimensional and {}-dimensional", self.to_str(), args[0].dimension(), args[1].dimension());
    //             }
    //         },
    //         _ => {},
    //     }
    // }

    // pub fn dimension(&self, args: &Vec<TypedExpression>) -> usize {
    //     self.check_args(&args);

    //     match self {
    //         Self::COS | Self::SIN | Self::TAN | Self::SQRT | Self::LOG | Self::EXP | Self::POW => {
    //             0
    //         },
    //         Self::CIS => {
    //             args[0].dimension()
    //         },
    //     }
    // }

    pub fn intrinsic(&self) -> Option<Intrinsic> {
        match self {
            Self::COS => Intrinsic::find("llvm.cos"),
            Self::SIN => Intrinsic::find("llvm.sin"),
            Self::SQRT => Intrinsic::find("llvm.sqrt"),
            Self::POW => Intrinsic::find("llvm.powi"),
        }
    }
}

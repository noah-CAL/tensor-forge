use std::fmt::{Debug, Display};

pub fn validate_error_messages(errors: &[impl Display + Debug]) {
    for err in errors {
        assert!(
            err.to_string().len() >= 5,
            "Display not implemented properly for {:?}",
            err
        );
    }
}

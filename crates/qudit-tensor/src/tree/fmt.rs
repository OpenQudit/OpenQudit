pub trait PrintTree {
    fn modify_prefix_for_child(
        &self,
        prefix: &str,
        last_child: bool,
    ) -> String {
        let mut new_prefix = String::from(prefix);

        // replace last ╠═ or ╚═ with ║
        new_prefix.pop();
        new_prefix.pop();
        new_prefix.pop();

        match new_prefix.pop() {
            Some('╠') => new_prefix.push_str("║   "),
            Some('╚') => new_prefix.push_str("    "),
            _ => (),
        };

        if last_child {
            new_prefix.push_str("╚══ ");
        } else {
            new_prefix.push_str("╠══ ");
        }

        new_prefix
    }

    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>);
}

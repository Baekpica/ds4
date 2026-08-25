/// C `DS4_SERVER_CONT_TOOLS_{ANTHROPIC,RESPONSES}` at v0.6.3-dfm.
/// Default ON; only the exact C string `"0"` forces off.
pub const fn cont_tools_from_env(
    anthropic: Option<&[u8]>,
    responses: Option<&[u8]>,
) -> (bool, bool) {
    (
        !matches!(anthropic, Some(b"0")),
        !matches!(responses, Some(b"0")),
    )
}

#[cfg(test)]
mod cont_tools_env_tests {
    use super::cont_tools_from_env;

    #[test]
    fn cont_tools_default_equals_c_when_env_unset() {
        // Given: C getenv() returns NULL for both kill switches
        // When: Rust applies the v0.6.3-dfm contract
        // Then: both surfaces stay ON
        assert_eq!(cont_tools_from_env(None, None), (true, true));
    }

    #[test]
    fn cont_tools_forced_off_when_env_is_zero() {
        // Given: exact C string "0" on each surface
        // When: parse
        // Then: that surface is off
        assert_eq!(cont_tools_from_env(Some(b"0"), Some(b"0")), (false, false));
    }

    #[test]
    fn cont_tools_env_zero_is_per_surface() {
        assert_eq!(cont_tools_from_env(Some(b"0"), None), (false, true));
        assert_eq!(cont_tools_from_env(None, Some(b"0")), (true, false));
    }
}

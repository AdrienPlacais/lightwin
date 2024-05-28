"""Test that the lines of the ``.dat`` are properly understood."""

from tracewin_utils.load import slice_dat_line


class TestSlice:
    """Test functions to convert a ``.dat`` line to list of arguments."""

    def test_basic_line(self) -> None:
        """Test that a basic line is properly sliced."""
        line = "DRIFT 76"
        expected = ["DRIFT", "76"]
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_line_with_more_arguments(self) -> None:
        line = "FIELD_MAP 100 5 0.9 0.7 54e4 3 65.6e10"
        expected = line.split()
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_basic_comment(self) -> None:
        """Test that a basic comment is properly sliced."""
        line = ";DRIFT 76"
        expected = [";", "DRIFT 76"]
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_basic_comment_with_space(self) -> None:
        """Test that a basic comment is properly sliced."""
        line = "; DRIFT 76"
        expected = [";", "DRIFT 76"]
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_element_with_a_name(self) -> None:
        """Test that a named element is properly sliced."""
        line = "Louise: DRIFT 76"
        expected = ["Louise", "DRIFT", "76"]
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_element_with_a_name_additional_space(self) -> None:
        """Test that a named element is properly sliced."""
        line = "Michel : DRIFT 76"
        expected = ["Michel", "DRIFT", "76"]
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_diagnostic_with_a_weight(self) -> None:
        """Test that a weighted element is properly sliced."""
        line = "DIAG_BONJOURE(1e3) 777 0 1 2"
        expected = ["DIAG_BONJOURE", "(1e3)", "777", "0", "1", "2"]
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_diagnostic_with_a_weight_additional_space(self) -> None:
        """Test that a weighted element is properly sliced."""
        line = "DIAG_BONJOURE (1e3) 777 0 1 2"
        expected = ["DIAG_BONJOURE", "(1e3)", "777", "0", "1", "2"]
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_diagnostic_with_a_weight_different_fmt(self) -> None:
        """Test that a weighted element is properly sliced."""
        line = "DIAG_BONJOURE (4.5) 777 0 1 2"
        expected = ["DIAG_BONJOURE", "(4.5)", "777", "0", "1", "2"]
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_multiple_semicommas(self) -> None:
        """Check that when we have several ;, only the first is kept."""
        line = ";;;;;;;; Section1: ;;;;;;;"
        expected = [line[0], line[1:]]
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_comment_at_end_of_line_is_removed(self) -> None:
        """Test that EOL comments are removed to avoid any clash."""
        line = "DRIFT 76 ; this drift is where we put the coffee machine"
        expected = ["DRIFT", "76"]
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_line_with_nothing_but_spaces(self) -> None:
        """Test that empty line is correctly understood."""
        line = "    "
        expected = []
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"
